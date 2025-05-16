import sys
sys.path.append('/home/anshsg13/Documents/honors_thesis/mambabcd_model/classification/models/')

from torch.optim.lr_scheduler import StepLR
import matplotlib
import argparse
import os
import time
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import multiprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau

from MambaCD.changedetection.configs.config import get_config
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.MambaBCD import STMambaBCD
from sklearn import metrics
from classification.models.vmamba import VSSBlock, EinFFT
import MambaCD.changedetection.utils_func.lovasz_loss as L
import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from joblib import Parallel, delayed



class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        weight = self.weight
        if weight is None:
            weight = torch.tensor([1, 2.3]).to(inputs.device)
        else:
            weight = weight.to(inputs.device)
        return F.cross_entropy(
            inputs,
            targets,
            weight=weight,
            reduction=self.reduction,
            ignore_index=2
        )




class FocalLoss(nn.Module):
    def __init__(self, gamma=4.0, alpha=.1, reduction='mean'):
        """
        Args:
            gamma (float): Focusing parameter for hard examples.
            alpha (float or list, optional): Weight factor for classes.
                For binary classification, a float (e.g., 0.75) means the positive class
                gets that weight and the negative class gets (1 - alpha).
            reduction (str): Specifies the reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1 - alpha])
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits, shape [N, C, H, W] where C=1 or 2.
            targets: Ground truth labels, shape [N, H, W] with values {0,1}.
        Returns:
            Focal loss value.
        """
        b, c, h, w = inputs.shape
        inputs = inputs.view(b, c, -1)  # [N, C, H*W]
        targets = targets.view(b, -1)   # [N, H*W]
        probs = F.softmax(inputs, dim=1)  # [N, 2, H*W]
        # One-hot encode targets to [N, 2, H*W]
        targets_one_hot = torch.zeros_like(probs)
        # Using scatter_ to one-hot encode: targets values should be 0 or 1.
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        eps = 1e-6
        log_p = torch.log(probs + eps)

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_factor = self.alpha.to(inputs.device)[None, :, None]  # shape [1, 2, 1]
            log_p = alpha_factor * log_p

        # Compute the probability of the true class.
        pt = (probs * targets_one_hot).sum(dim=1, keepdim=True)  # [N, 1, H*W]
        focal_factor = (1 - pt) ** self.gamma

        loss = - focal_factor * log_p * targets_one_hot
        loss = loss.sum(dim=1)  # sum over classes -> [N, H*W]
        loss = loss.mean(dim=1) # mean over pixels -> [N]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class Trainer(object):
    def __init__(self, args, train_data_loader=None):
        self.args = args
        config = get_config(args)
        self.max_epochs = args.max_epochs  

        # Train loader
        if train_data_loader is None:
            self.train_data_loader = make_data_loader(args, phase='train')
        else:
            self.train_data_loader = train_data_loader
        self.val_data_loader = make_data_loader(args, phase='val')
        self.test_data_loader = make_data_loader(args, phase='test')
        print(f"train samples this epoch: {len(self.train_data_loader.dataset)}")
        print(f" val samples this epoch: {len(self.val_data_loader.dataset)}")


        self.evaluator = Evaluator(num_class=2)
        self.train_losses = []
        self.val_losses   = []
        self.grad_norms = []

        self.deep_model = STMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        device = torch.device('cuda:0')
        self.deep_model = self.deep_model.cuda()
        #print("model params", self.deep_model)

        ngpus = torch.cuda.device_count()
        if ngpus > 1:
            print(f"Using DataParallel on {ngpus} GPUs...")
            self.deep_model = nn.DataParallel(self.deep_model)
        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optim,
            mode='min',           # we want to minimize val‐loss
            factor=0.5,           # multiply LR by 0.5 on plateau
            patience=2,           # wait 2 epochs without improvement
            min_lr=1e-6,          # lower bound on LR
            verbose=True
        )

        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        # Add early stopping parameters
        self.patience = 10  # Number of epochs to wait for improvement
        self.min_delta = 0.001  # Minimum change to qualify as improvement
        self.best_val_loss = float('inf')  # Initialize with infinity
        self.wait_epochs = 0  # Counter for epochs without improvement
        self.early_stop = False  # Flag to indicate if training should stop
        self.best_model_state = None  # Store best model weights

    # NEW: Training method in epoch style (no iteration-level plotting)
    def training(self, loss_fn):
        """
        Full training loop for `max_epochs`.
        Each epoch we:
        1) Compute average training loss
        2) Compute average validation loss
        3) Save or plot results
        4) Display the estimated time remaining
        """
        best_precision = 0.0
        best_f1 = 0.0
        best_round = []
        epoch_durations = []  # list to record each epoch's duration

        for epoch in range(self.max_epochs):
            epoch_start = time.time()  # Start time of the epoch
            print(f"\n--- Starting Epoch {epoch+1}/{self.max_epochs} ---")

            # 1) Training Loop
            self.deep_model.train()
            epoch_train_loss_sum = 0.0
            epoch_train_samples = 0

            for batch_idx, data in enumerate(self.train_data_loader):
                pre_imgs, post_imgs, labels, _ = data
                pre_imgs = pre_imgs.cuda(0).float()
                post_imgs = post_imgs.cuda(0)
                labels = labels.cuda(0).long()


                outputs = self.deep_model(pre_imgs, post_imgs)
                self.optim.zero_grad()

                loss_per_pixel = loss_fn(outputs, labels)
                valid_mask = (labels != 2).float()
                final_loss = (loss_per_pixel * valid_mask).sum() / (valid_mask.sum() + 1e-6)

                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), max_norm=5.0)
                total_norm = 0.0
                for param in self.deep_model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.grad_norms.append(total_norm)
                self.optim.step()

                bs = pre_imgs.size(0)
                epoch_train_loss_sum += final_loss.item() * bs
                epoch_train_samples += bs

            epoch_train_loss = epoch_train_loss_sum / max(epoch_train_samples, 1)
            self.train_losses.append(epoch_train_loss)

            # 2) Validation Loop
            val_loss = self._compute_val_loss(loss_fn)
            self.scheduler.step(val_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss - self.min_delta:
                # Improved performance
                self.best_val_loss = val_loss
                self.wait_epochs = 0
                # Store the best model weights
                self.best_model_state = self.deep_model.state_dict().copy()
                print(f"Validation loss improved to {val_loss:.4f}")
            else:
                # No improvement
                self.wait_epochs += 1
                print(f"No improvement in validation loss for {self.wait_epochs} epochs. Best: {self.best_val_loss:.4f}")
                if self.wait_epochs >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    self.early_stop = True


            # 3) Record elapsed time and estimate remaining time
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            epoch_durations.append(epoch_duration)
            avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
            epochs_left = self.max_epochs - (epoch + 1)
            eta = epochs_left * avg_epoch_time

            # 4) Print summary along with elapsed time and estimated time left
            print(f"Epoch {epoch+1} Summary: Train Loss={epoch_train_loss:.4f}, Val Loss={val_loss:.4f}")
            print(f"Epoch Duration: {epoch_duration:.2f} sec, Estimated Time Remaining: {eta:.2f} sec")

            # 5) Optionally, check accuracy on validation set
            self.deep_model.eval()    
            rec, pre, oa, f1_score, iou, kc = self.validation()  # get some metrics
            if f1_score > best_f1:
                best_f1 = f1_score
                best_round = [rec, pre, oa, f1_score, iou, kc]
                model_path = os.path.join(self.model_save_path, f"best_model_epoch{epoch+1}.pth")
                torch.save(self.deep_model.state_dict(), model_path)
                print(f"New best model saved to {model_path}")

            # Check if we should stop early
            if self.early_stop:
                print("Early stopping triggered!")
                # Load the best model weights
                if self.best_model_state is not None:
                    self.deep_model.load_state_dict(self.best_model_state)
                    print("Loaded best model weights from epoch with lowest validation loss")
                break


        print("\nTraining complete.")
        print("Best round metrics: ", best_round)
        self._plot_train_val_losses()
        self._plot_grad_norms()


    def _plot_grad_norms(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.grad_norms, label='Gradient Norm')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms During Training')
        plt.legend()
        plt.grid(True)
        
        grad_norm_path = os.path.join(self.model_save_path, 'grad_norms.png')
        plt.savefig(grad_norm_path)
        plt.close()
        print(f"Gradient Norm curve saved to {grad_norm_path}")

    def _compute_val_loss(self, loss_fn):
        self.deep_model.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for data in self.val_data_loader:
                pre_imgs, post_imgs, labels, _ = data
                pre_imgs = pre_imgs.cuda().float()
                post_imgs = post_imgs.cuda()
                labels = labels.cuda().long()

                outputs = self.deep_model(pre_imgs, post_imgs)
                loss_per_pixel = loss_fn(outputs, labels)
                valid_mask = (labels != 2).float()
                final_val_loss = (loss_per_pixel * valid_mask).sum() / (valid_mask.sum() + 1e-6)

                bs = pre_imgs.size(0)
                val_loss_sum += final_val_loss.item() * bs
                val_samples  += bs

        return val_loss_sum / max(val_samples, 1)

    def _plot_train_val_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses,   label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        plt.legend()
        plt.grid(True)

        out_path = os.path.join(self.model_save_path, 'train_val_loss_curve.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Train vs Val Loss curve saved to {out_path}")

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()

        # If you only want to compute metrics on test_data_path, keep this loader
        # Or if you want on the val_data_loader, choose which you prefer
        
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for data in self.val_data_loader:
                pre_change_imgs, post_change_imgs, labels, _ = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)
                output_1 = output_1.data.cpu().numpy() 
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()

                valid_mask = (labels != 2)
                valid_labels = labels[valid_mask]
                valid_preds  = output_1[valid_mask]

                self.evaluator.add_batch(valid_labels, valid_preds)
                
        f1_score = self.evaluator.Pixel_F1_score()
        f05 = self.evaluator.Pixel_F05_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Recall rate={rec}, Precision={pre}, OA={oa}, F1={f1_score}, F05={f05}, IoU={iou}, Kappa={kc}')
        return rec, pre, oa, f1_score, iou, kc


    def final_validation(self):
        print('Starting final evaluation...')
        
        # ----------- Validation: Calibration Curve -----------
        val_probs, val_labels = [], []
        # Loop over validation data
        with torch.no_grad():
            for idx, data in enumerate(self.val_data_loader):
                pre_imgs, post_imgs, labels, _ = data
                #print(f"Test batch {idx} has unique labels: {torch.unique(labels)}")
                pre_imgs = pre_imgs.cuda().float()
                post_imgs = post_imgs.cuda()
                labels = labels.cuda().long()
                
                # Get model output (softmax probabilities already computed in inference mode)
                output = self.deep_model(pre_imgs, post_imgs, inference=True)
                # For each pixel, use the probability for class 1 (changed)
                prob_map = output[:, 1, :, :] if output.shape[1] > 1 else output[:, 0, :, :]
                
                val_probs.append(prob_map.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        
        # Flatten the validation predictions and ground truths
        flat_val_probs = np.concatenate([p.flatten() for p in val_probs])
        flat_val_labels = np.concatenate([l.flatten() for l in val_labels])
        
        # Compute calibration curve (reliability diagram)
        valid = flat_val_labels != 2
        flat_val_labels = flat_val_labels[valid]
        flat_val_probs = flat_val_probs[valid]
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(flat_val_labels, flat_val_probs, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_pred, frac_pos, "s-", label="Calibration Curve")
        plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Validation Calibration Curve")
        plt.legend()
        calib_path = os.path.join(self.model_save_path, 'calibration_curve.png')
        plt.savefig(calib_path)
        plt.close()
        print("Calibration curve saved to", calib_path)
        
        # ----------- Optimal Threshold: Precision-Recall Analysis -----------
        from sklearn.metrics import precision_recall_curve
        # Compute precision, recall and thresholds from the precision-recall curve.
        precision, recall, pr_thresholds = precision_recall_curve(flat_val_labels, flat_val_probs)
        
        # Compute the F0.5 score for each threshold to prioritize precision.
        # Note: pr_thresholds has one fewer element than precision and recall.
        beta = 0.5
        f05_scores = (1 + beta**2) * (precision[:-1] * recall[:-1]) / (beta**2 * precision[:-1] + recall[:-1] + 1e-8)
        
        optimal_idx = np.argmax(f05_scores)
        optimal_threshold = pr_thresholds[optimal_idx]
        print("Optimal threshold from Precision-Recall analysis (F0.5):", optimal_threshold)
        
        # ----------- Test: Final Metrics -----------
        self.evaluator.reset()
        sample_pre, sample_prob = None, None
        with torch.no_grad():
            for idx, data in enumerate(self.test_data_loader):
                pre_imgs, post_imgs, labels, _ = data
                pre_imgs = pre_imgs.cuda().float()
                post_imgs = post_imgs.cuda()
                labels = labels.cuda().long()
                
                output = self.deep_model(pre_imgs, post_imgs, inference=True)
                prob_map = output[:, 1, :, :] if output.shape[1] > 1 else output[:, 0, :, :]
                # Generate binary prediction using the optimal threshold.
                binary_map = (prob_map > optimal_threshold).type(torch.uint8)
                labels_np = labels.cpu().numpy()
                binary_map_np = binary_map.cpu().numpy()
                valid_mask = (labels_np != 2)
                valid_labels = labels_np[valid_mask]
                valid_preds  = binary_map_np[valid_mask]
                
                self.evaluator.add_batch(valid_labels, valid_preds)
                
                if idx == 0:
                    sample_pre = pre_imgs.cpu().numpy()[0]
                    sample_prob = prob_map.cpu().numpy()[0]
        
        # Get final evaluation metrics.
        f1_score = self.evaluator.Pixel_F1_score()
        f05 = self.evaluator.Pixel_F05_score()
        oa       = self.evaluator.Pixel_Accuracy()
        rec      = self.evaluator.Pixel_Recall_Rate()
        prec     = self.evaluator.Pixel_Precision_Rate()
        iou      = self.evaluator.Intersection_over_Union()
        kc       = self.evaluator.Kappa_coefficient()
        print(f'Final Evaluation - Recall: {rec}, Precision: {prec}, OA: {oa}, F1: {f1_score}, F05: {f05}, IoU: {iou}, Kappa: {kc}')
        
        return rec, sample_pre, oa, f1_score, iou, kc



import pandas as pd

def objective(trial, args, base_train_dataloader, weighted_loss):
    """Efficient 2-epoch objective function with focused parameter ranges"""
    from MambaCD.changedetection.datasets.make_data_loader import make_data_loader
    
    # --- GPU assignment for parallel processing ---
    # --- GPU assignment for parallel processing ---
    # Alternative approach without get_worker_id()
    # Each trial selects a GPU based on its number
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        # When running in parallel, each process will have a different trial number
        gpu_id = trial.number % num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"Trial {trial.number} running on GPU {gpu_id}")
    else:
        gpu_id = 0
        torch.cuda.set_device(gpu_id)
    print(f"Trial {trial.number} running on GPU {gpu_id}")
    
    # --- Track trial start time ---
    start_time = time.time()
    
    # --- Tune model hyperparameters with logarithmic scales for better exploration ---
    # For sparsity_threshold, use log scale to better explore small values
    sparsity_threshold = trial.suggest_float("sparsity_threshold", 0.001, 0.1, log=True)
    
    # For scale, also use log scale
    scale = trial.suggest_float("scale", 0.001, 0.1, log=True)
    
    # --- Tune loss function weights ---
    loss_weight0 = trial.suggest_float("loss_weight0", 0.1, 1.0, step=0.1)
    loss_weight1 = trial.suggest_float("loss_weight1", 1.0, 7.0, step=0.1)
    
    # --- Fixed learning rate and batch size ---
    learning_rate = 0.0001  # Close to your original value
    
    # --- Create training dataloader ---
    train_dataloader = make_data_loader(args)
    
    # --- Create loss function with dynamic weights ---
    loss_fn = WeightedCrossEntropyLoss(
        weight=torch.tensor([loss_weight0, loss_weight1]),
        reduction='none'
    ).cuda()
    
    # --- Use exactly 2 epochs for all trials ---
    trial_epochs = 2
    
    # --- Create Trainer using the dataloader ---
    trainer = Trainer(args, train_data_loader=train_dataloader)
    trainer.max_epochs = trial_epochs
    
    # We'll skip DataParallel here since we're doing trial-level parallelism
    # Each trial runs on a single GPU, but we run multiple trials in parallel
    
    # Update the optimizer learning rate
    for param_group in trainer.optim.param_groups:
        param_group['lr'] = learning_rate
    
    # --- Update model modules with tuned hyperparameters ---
    for layer in trainer.deep_model.modules():
        if isinstance(layer, VSSBlock) and hasattr(layer, "mlp") and isinstance(layer.mlp, EinFFT):
            layer.mlp.sparsity_threshold = sparsity_threshold
            layer.mlp.scale = scale
    
    # --- Use learning rate warmup for faster initial learning ---
    warmup_fraction = 0.1  # Warm up for 10% of the first epoch
    total_steps = len(train_dataloader) * trial_epochs
    warmup_steps = int(total_steps * warmup_fraction)
    
    # --- Training Loop with warmup ---
    global_step = 0
    epoch_f1_scores = []
    
    for epoch in range(trial_epochs):
        trainer.deep_model.train()
        running_loss = 0.0
        
        for batch_idx, data in enumerate(trainer.train_data_loader):
            # Apply learning rate warmup
            if global_step < warmup_steps:
                warmup_factor = global_step / warmup_steps
                current_lr = learning_rate * warmup_factor
                for param_group in trainer.optim.param_groups:
                    param_group['lr'] = current_lr
            
            pre_change_imgs, post_change_imgs, labels, _ = data
            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()
            
            output = trainer.deep_model(pre_change_imgs, post_change_imgs)
            trainer.optim.zero_grad()
            current_loss = loss_fn(output, labels)
            final_loss = current_loss.mean()
            running_loss += final_loss.item()
            final_loss.backward()
            trainer.optim.step()
            
            global_step += 1
        
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(trainer.train_data_loader)
        
        # Evaluate after each epoch
        trainer.deep_model.eval()
        _, _, _, f1_score, _, _ = trainer.validation()
        epoch_f1_scores.append(f1_score)
        
        # Track metrics for analysis
        trial.report(f1_score, epoch)
        trial.set_user_attr(f"loss_epoch_{epoch}", avg_loss)
        trial.set_user_attr(f"f1_epoch_{epoch}", f1_score)
        print(f"Trial {trial.number} on GPU {gpu_id}, Epoch {epoch}: F1={f1_score:.4f}, Loss={avg_loss:.4f}")
        
        # Check for pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # --- Calculate improvement from epoch 1 to epoch 2 ---
    if len(epoch_f1_scores) >= 2:
        improvement = epoch_f1_scores[1] - epoch_f1_scores[0]
        trial.set_user_attr("improvement", improvement)
    
    # --- Final evaluation ---
    trainer.deep_model.eval()
    _, _, _, final_f1, _, _ = trainer.validation()
    
    # Record trial duration
    duration = time.time() - start_time
    trial.set_user_attr("duration", duration)
    
    # Use a weighted sum of final F1 and improvement as objective
    # This helps select parameters that continue to improve with more training
    learning_potential = final_f1 + (0.2 * improvement if len(epoch_f1_scores) >= 2 else 0)
    trial.set_user_attr("learning_potential", learning_potential)
    
    print(f"Trial {trial.number} completed on GPU {gpu_id}: Final F1={final_f1:.4f}, Duration={duration:.1f}s")
    
    return learning_potential

def validate_top_configurations(args, base_train_dataloader, weighted_loss, top_trials, n_epochs=5):
    """Validate top configurations with more epochs to verify performance"""
    results = []
    
    for i, trial in enumerate(top_trials):
        # Distribute validation across available GPUs
        gpu_id = i % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
        print(f"Validating trial {trial.number} on GPU {gpu_id}")
        
        params = trial.params
        
        # --- Create training dataloader ---
        train_dataloader = make_data_loader(args)
        
        # --- Create loss function with trial weights ---
        loss_fn = WeightedCrossEntropyLoss(
            weight=torch.tensor([params["loss_weight0"], params["loss_weight1"]]),
            reduction='none'
        ).cuda()
        
        # --- Create Trainer using the dataloader ---
        trainer = Trainer(args, train_data_loader=train_dataloader)
        trainer.max_epochs = n_epochs
        
        # --- Update model modules with tuned hyperparameters ---
        for layer in trainer.deep_model.modules():
            if isinstance(layer, VSSBlock) and hasattr(layer, "mlp") and isinstance(layer.mlp, EinFFT):
                layer.mlp.sparsity_threshold = params["sparsity_threshold"]
                layer.mlp.scale = params["scale"]
        
        # --- Training Loop for more epochs ---
        trainer.deep_model.train()
        for epoch in range(n_epochs):
            for batch_idx, data in enumerate(trainer.train_data_loader):
                pre_change_imgs, post_change_imgs, labels, _ = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()
                
                output = trainer.deep_model(pre_change_imgs, post_change_imgs)
                trainer.optim.zero_grad()
                current_loss = loss_fn(output, labels)
                final_loss = current_loss.mean()
                final_loss.backward()
                trainer.optim.step()
            
            # Print progress
            print(f"Validation of trial {trial.number} on GPU {gpu_id}: Epoch {epoch+1}/{n_epochs} completed")
        
        # --- Final evaluation ---
        trainer.deep_model.eval()
        _, _, _, final_f1, _, _ = trainer.validation()
        
        results.append({
            "params": params,
            "final_f1": final_f1,
            "trial_number": trial.number
        })
        
        print(f"Validated Trial {trial.number} with {n_epochs} epochs on GPU {gpu_id}: F1={final_f1:.4f}")
    
    return results

def visualize_results(study, validation_results=None):
    """Create comprehensive visualizations of the optimization process"""
    # Get trial data
    trials_df = study.trials_dataframe()
    
    # 1. Optimization history plot
    plt.figure(figsize=(10, 6))
    plt.scatter(trials_df['number'], trials_df['value'], alpha=0.7)
    plt.xlabel("Trial Number")
    plt.ylabel("Objective Value")
    plt.title("Hyperparameter Optimization History")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("optimization_history.png")
    plt.close()
    
    # 2. Parameter importance plot (if available)
    try:
        from optuna.visualization import plot_param_importances
        fig = plot_param_importances(study)
        fig.write_image("parameter_importance.png")
    except Exception as e:
        print(f"Could not create parameter importance plot: {e}")
    
    # 3. Correlation between parameters and objective
    plt.figure(figsize=(16, 12))
    
    # Get the parameter names
    param_names = ['sparsity_threshold', 'scale', 'loss_weight0', 'loss_weight1']
    
    # Create subplots
    for i, param in enumerate(param_names):
        if param in trials_df.columns:
            plt.subplot(2, 2, i+1)
            plt.scatter(trials_df[param], trials_df['value'], alpha=0.7)
            plt.xlabel(param)
            plt.ylabel("Objective Value")
            plt.title(f"Effect of {param}")
            plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("parameter_correlation.png")
    plt.close()
    
    # 4. Parallel coordinate plot (if available)
    try:
        from optuna.visualization import plot_parallel_coordinate
        fig = plot_parallel_coordinate(study)
        fig.write_image("parameter_relationships.png")
    except Exception as e:
        print(f"Could not create parallel coordinate plot: {e}")
    
    # 5. If validation results are available, plot comparison
    if validation_results:
        val_df = pd.DataFrame(validation_results)
        
        # Get the original 2-epoch F1 scores for these trials
        original_f1 = []
        for result in validation_results:
            trial_number = result['trial_number']
            trial = study.trials[trial_number]
            original_f1.append(trial.user_attrs.get('f1_epoch_1', 0))  # F1 after 2 epochs
        
        val_df['original_f1'] = original_f1
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        trial_numbers = val_df['trial_number'].tolist()
        
        x = np.arange(len(trial_numbers))
        width = 0.35
        
        plt.bar(x - width/2, val_df['original_f1'], width, label='2-Epoch F1')
        plt.bar(x + width/2, val_df['final_f1'], width, label='Extended Training F1')
        
        plt.xlabel("Trial Number")
        plt.ylabel("F1 Score")
        plt.title("Validation: 2-Epoch vs Extended Training")
        plt.xticks(x, trial_numbers)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("validation_comparison.png")
        plt.close()
        
        # Return the best validated configuration
        best_idx = val_df['final_f1'].idxmax()
        return val_df.iloc[best_idx]['params']
    
    return None

def run_hyperparameter_optimization(args):
    """Main function to run the hyperparameter optimization process"""
    from MambaCD.changedetection.datasets.make_data_loader import make_data_loader
    
    # --- Detect available GPUs ---
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available for parallel optimization")
    
    if num_gpus == 0:
        raise RuntimeError("No GPUs found. This script requires GPUs for training.")
    
    # --- Create the base train DataLoader ---
    base_train_dataloader = make_data_loader(args)
    
    # --- Create the weighted loss function ---
    weighted_loss = WeightedCrossEntropyLoss(reduction='none').cuda()
    
    # --- Initialize and configure the study ---
    # Use TPESampler for more efficient exploration
    sampler = TPESampler(n_startup_trials=7)  # Start with random sampling to build knowledge
    
    # Use MedianPruner to prune trials that fall below median at same step
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
 
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="mamba_hyperparameter_optimization",
        load_if_exists=True
    )
    
    print("Starting hyperparameter optimization...")
    n_trials = args.n_trials
    
    # Run the optimization with parallel execution
    # n_jobs = number of parallel trials to run (use the number of GPUs)
    # Note: This requires that the objective function handles GPU assignment
    study.optimize(
        lambda trial: objective(trial, args, base_train_dataloader, weighted_loss),
        n_trials=n_trials, 
        n_jobs=num_gpus,  # Run trials in parallel, one per GPU
        gc_after_trial=True  # Clean up memory after each trial
    )
    
    print("\nHyperparameter optimization completed!")
    print(f"Number of completed trials: {len(study.trials)}")
    
    # Filter out pruned trials for better analysis
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Number of completed trials (not pruned): {len(completed_trials)}")
    
    if completed_trials:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.4f}")
        print("\nBest hyperparameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        
        # Visualize the results
        print("\nGenerating visualization plots...")
        
        # Select top 3 trials for validation with more epochs
        print("\nValidating top configurations with more epochs...")
        top_trials = sorted(completed_trials, 
                           key=lambda t: t.value if t.value is not None else float('-inf'), 
                           reverse=True)[:min(3, len(completed_trials))]
        
        validation_results = validate_top_configurations(
            args, base_train_dataloader, weighted_loss, top_trials, n_epochs=5
        )
        
        # Visualize with validation results
        best_params = visualize_results(study, validation_results)
        if best_params:
            print("\nBest validated parameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            return best_params
        else:
            return study.best_params
    else:
        print("No trials completed successfully. Try adjusting parameters or reducing pruning aggressiveness.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for MambaCD")
    
    # Define arguments (same as your original code)
    parser.add_argument('--max_epochs', type=int, default=5, help="Number of epochs for final training")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='SYSU')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/train')
    parser.add_argument('--train_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--val_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--val_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--deforestation_thr', type=float, default=.0000000001)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument(
        "--balance_deforestation",     # True  → BalancedChangeDetectionDataset
        action="store_true",           # False → ChangeDetectionDatset (all chips)
        help="If set, the train/val loaders will emit a 50‑50 mix of "
            "deforested / non‑deforested chips. Otherwise they iterate "
            "over the full dataset in the natural proportion."
    )
    
    args = parser.parse_args()
    
    # Load data name lists
    with open(args.train_data_list_path, "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list
    
    if os.path.exists(args.test_data_list_path):
        print(f"File exists: {args.test_data_list_path}")
        with open(args.test_data_list_path, "r") as f:
            test_data_name_list = [data_name.strip() for data_name in f]
        args.test_data_name_list = test_data_name_list
        print(f"test_data_name_list first 5 entries: {test_data_name_list[:5]}")
    else:
        print(f"File does not exist: {args.test_data_list_path}")
    
    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [data_name.strip() for data_name in f]
    args.val_data_name_list = val_data_name_list
    
    # Run hyperparameter optimization
    #best_params = run_hyperparameter_optimization(args)
    best_params = None

    # --- Train final model with best parameters ---
    print("\n==== Training final model with best parameters ====")
    
    # Create trainer with best parameters
    trainer = Trainer(args)
    
    # Update model parameters
    for layer in trainer.deep_model.modules():
        if isinstance(layer, VSSBlock) and hasattr(layer, "mlp") and isinstance(layer.mlp, EinFFT):
            layer.mlp.sparsity_threshold = 0.09508911964533545
            layer.mlp.scale = 0.0011906415270652637
    
    # Update loss function
    final_loss_fn = WeightedCrossEntropyLoss(
        weight=torch.tensor([1, 5]).float(),
        reduction='none'
    ).cuda()
    
    # Train with the full number of epochs
    trainer.training(loss_fn=final_loss_fn)
    trainer.final_validation()
    

if __name__ == "__main__":
    main()

