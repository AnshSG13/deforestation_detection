#test20
import sys
sys.path.append('/home/anshsg13/Documents/honors_thesis/mambabcd_model/classification/models/')

from torch.optim.lr_scheduler import StepLR
import matplotlib
import argparse
import os
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("Agg")

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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
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
    def __init__(self, args, max_iters_for_trial=None, train_data_loader=None):
        self.args = args
        config = get_config(args)
        self.max_iters = max_iters_for_trial if max_iters_for_trial is not None else args.max_iters
        self.losses = []   # To store loss values over iterations
        self.iterations = []

        # Use provided DataLoader if available
        if train_data_loader is None:
            self.train_data_loader = make_data_loader(args)
        else:
            self.train_data_loader = train_data_loader

        #distribution = self.compute_class_distribution(self.train_data_loader)
        #print("Training label distribution:", distribution)

        self.evaluator = Evaluator(num_class=2)

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
        self.deep_model = self.deep_model.cuda()

        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.best_model_path = None
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)
            self.best_model_path = args.resume

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        #self.scheduler = StepLR(self.optim, step_size=1000, gamma=0.1)
        
        # Create evaluation DataLoaders once to avoid repeated creation
        self.test_dataset = ChangeDetectionDatset(self.args.test_dataset_path,
                                                   self.args.test_data_name_list,
                                                   self.args.crop_size, None, 'test')
        self.test_data_loader = DataLoader(self.test_dataset,
                                           batch_size=16,
                                           num_workers=1,
                                           drop_last=False,
                                           persistent_workers=True)
        
        self.val_dataset = ChangeDetectionDatset(self.args.val_dataset_path,
                                                  self.args.val_data_name_list,
                                                  self.args.crop_size, None, 'val')
        self.val_data_loader = DataLoader(self.val_dataset,
                                          batch_size=16,
                                          num_workers=1,
                                          drop_last=False,
                                          persistent_workers=True)

    def compute_class_distribution(self, dataloader):
        all_labels = []
        for batch in dataloader:
            _, _, labels, _ = batch  
            all_labels.append(labels.cpu().numpy())
        all_labels = np.concatenate(all_labels, axis=None)
        unique, counts = np.unique(all_labels, return_counts=True)
        return dict(zip(unique, counts))

    def save_loss_curve(self):
        """
        Plot and save the loss curve based on collected losses during training.
        Uses only PyTorch and matplotlib.
        """
        if len(self.losses) == 0:
            print("No loss values found to plot.")
            return
        
        # Convert lists to tensors for easier manipulation
        losses_tensor = torch.tensor(self.losses)
        iterations_tensor = torch.tensor(self.iterations)
        
        # Create figure using matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(iterations_tensor.cpu().numpy(), losses_tensor.cpu().numpy())
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        
        # Save the figure
        loss_curve_path = os.path.join(self.model_save_path, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"Loss curve saved to {loss_curve_path}")
        
        # Save the raw tensor data for future use
        loss_data_path = os.path.join(self.model_save_path, 'loss_data.pt')
        torch.save({
            'iterations': iterations_tensor,
            'losses': losses_tensor
        }, loss_data_path)
        print(f"Loss data saved to {loss_data_path}")
    
    def training(self, loss):
        best_kc = 0.0
        best_precision = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)

        self.losses = []
        self.iterations = []
        
        # Lists for storing gradient norm data
        gradient_norms = []
        gradient_iters = []

        for _ in tqdm(range(elem_num)):
            itera, data = next(train_enumerator)
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            output_1 = self.deep_model(pre_change_imgs, post_change_imgs)
            self.optim.zero_grad()
            #print("Model output shape:", output_1.shape)

            final_loss = loss(output_1, labels)
            final_loss.backward()
            
            # Compute total gradient norm over all parameters
            total_norm = 0.0
            for name, param in self.deep_model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)
            gradient_iters.append(itera)

            self.optim.step()
            #self.scheduler.step()

            self.losses.append(final_loss.item())
            self.iterations.append(itera)
            if (itera + 1) % 10 == 0:
                print(f'Iteration {itera + 1}, loss: {final_loss.item():.4f}, grad norm: {total_norm:.4f}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    if pre > best_precision:
                        torch.save(self.deep_model.state_dict(),
                                os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        best_precision = pre
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

        self.save_loss_curve()
        
        # Plot and save gradient norms chart
        plt.figure(figsize=(10, 6))
        plt.plot(gradient_iters, gradient_norms, label='Gradient Norm')
        plt.xlabel('Iteration')
        plt.ylabel('Total Gradient Norm')
        plt.title('Gradient Norms over Training Iterations')
        plt.legend()
        grad_norm_path = os.path.join(self.model_save_path, 'grad_norms.png')
        plt.savefig(grad_norm_path)
        plt.close()
        print(f"Gradient norms chart saved to {grad_norm_path}")

        print('The accuracy of the best round is ', best_round)
        self.final_validation()

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset = ChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=16, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels, _ = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()

                self.evaluator.add_batch(labels, output_1)
                
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Recall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc

    def final_validation(self):
        print('Starting final evaluation...')
        
        # ----------- Validation: Calibration Curve -----------
        val_probs, val_labels = [], []
        # Loop over validation data
        with torch.no_grad():
            for data in self.val_data_loader:
                pre_imgs, post_imgs, labels, _ = data
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
                self.evaluator.add_batch(labels.cpu().numpy(), binary_map.cpu().numpy())
                
                if idx == 0:
                    sample_pre = pre_imgs.cpu().numpy()[0]
                    sample_prob = prob_map.cpu().numpy()[0]
        
        # Get final evaluation metrics.
        f1_score = self.evaluator.Pixel_F1_score()
        oa       = self.evaluator.Pixel_Accuracy()
        rec      = self.evaluator.Pixel_Recall_Rate()
        prec     = self.evaluator.Pixel_Precision_Rate()
        iou      = self.evaluator.Intersection_over_Union()
        kc       = self.evaluator.Kappa_coefficient()
        print(f'Final Evaluation - Recall: {rec}, Precision: {prec}, OA: {oa}, F1: {f1_score}, IoU: {iou}, Kappa: {kc}')
        
        # Optionally, save a sample visualization.
        if sample_pre is not None and sample_prob is not None:
            sample_binary = (sample_prob > optimal_threshold).astype(np.uint8)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            if sample_pre.shape[0] == 3:
                axes[0].imshow(sample_pre.transpose(1, 2, 0))
            else:
                axes[0].imshow(sample_pre.squeeze(), cmap="gray")
            axes[0].set_title("Input Image (Pre-change)")
            axes[0].axis('off')
            
            im1 = axes[1].imshow(sample_prob, cmap="viridis")
            axes[1].set_title("Probability Map")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
            
            axes[2].imshow(sample_binary, cmap="gray")
            axes[2].set_title("Thresholded Output")
            axes[2].axis('off')
            
            sample_output_path = os.path.join(self.model_save_path, 'threshold_output.png')
            plt.savefig(sample_output_path)
            plt.close()
            print("Sample output saved to", sample_output_path)
        
        return rec, sample_pre, oa, f1_score, iou, kc


def objective(trial, args, train_dataloader, loss):
    # Hyperparameter tuning
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    smooth = trial.suggest_float("smooth", 0.5, 2.0)
    sparsity_threshold = trial.suggest_float("sparsity_threshold", 0.001, 0.1)
    scale = trial.suggest_float("scale", 0.01, 0.1)
    
    # Create Trainer using the preloaded DataLoader
    trainer = Trainer(args, max_iters_for_trial=5000, train_data_loader=train_dataloader)
    
    # Update model modules with tuned hyperparameters
    for layer in trainer.deep_model.modules():
        if isinstance(layer, VSSBlock) and hasattr(layer, "mlp") and isinstance(layer.mlp, EinFFT):
            layer.mlp.sparsity_threshold = sparsity_threshold
            layer.mlp.scale = scale

    tversky_loss = FocalLoss(alpha=alpha, gamma=gamma, smooth=smooth).cuda()
    trainer.deep_model.train()
    train_enumerator = enumerate(trainer.train_data_loader)
    total_iters = trainer.max_iters
    eval_freq = 50
    for iteration in range(total_iters):
        try:
            _, data = next(train_enumerator)
        except StopIteration:
            train_enumerator = enumerate(trainer.train_data_loader)
            _, data = next(train_enumerator)

        pre_change_imgs, post_change_imgs, labels, _ = data
        pre_change_imgs = pre_change_imgs.cuda().float()
        post_change_imgs = post_change_imgs.cuda()
        labels = labels.cuda().long()

        output = trainer.deep_model(pre_change_imgs, post_change_imgs)
        trainer.optim.zero_grad()
        loss = loss(output, labels)
        loss.backward()
        trainer.optim.step()
        trainer.losses.append(loss.item())
        trainer.iterations.append(iteration)

        if (iteration + 1) % eval_freq == 0:
            trainer.deep_model.eval()
            _, _, _, f1_score, _, _ = trainer.validation_regular()
            trainer.deep_model.train()
            trial.report(f1_score, iteration)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    trainer.deep_model.eval()
    _, _, _, final_f1, _, _ = trainer.validation_regular()
    return final_f1


def main():
    parser = argparse.ArgumentParser(description="Training on DEFORESTATION dataset")
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
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_trials', type=int, default=5, help='Number of Optuna trials')

    args = parser.parse_args()
    
    with open(args.train_data_list_path, "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [data_name.strip() for data_name in f]
    args.val_data_name_list = val_data_name_list

    
    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: objective(trial, args, train_dataloader), n_trials=args.n_trials)
    # print("Best hyperparameters:", study.best_params)
    # best_params = study.best_params

    # best_params = {
    #     "alpha": 0.5,
    #     "gamma": 2.0,
    #     "smooth": 1.0,
    #     "sparsity_threshold": 0.01,
    #     "scale": 0.05
    # }
    # Train final model with best hyperparameters using the same training DataLoader
    focal_loss = FocalLoss(gamma=2.0, alpha=0.1, reduction='mean').cuda()
    trainer = Trainer(args)
    # for layer in trainer.deep_model.modules():
    #     if isinstance(layer, VSSBlock) and hasattr(layer, "mlp") and isinstance(layer.mlp, EinFFT):
    #         layer.mlp.sparsity_threshold = best_params["sparsity_threshold"]
    #         layer.mlp.scale = best_params["scale"]

    # hybrid_loss = HybridLoss(alpha=best_params["alpha"],
    #                          gamma=best_params["gamma"],
    #                          smooth=best_params["smooth"]).cuda()
    trainer.training(loss=focal_loss)


if __name__ == "__main__":
    main()
