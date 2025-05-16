import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.networks import *

import torch
import torch.optim as optim
import numpy as np
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, softmax_helper, FocalLoss, mIoULoss, mmIoULoss

from misc.logger_tool import Logger, Timer

from utils import de_norm

from tqdm import tqdm

# For calibration curve
from sklearn.calibration import calibration_curve

class CDTrainer():

    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders = dataloaders

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device(
            "cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0 else "cpu"
        )
        print(self.device)
        # Move network to the device
        self.net_G = self.net_G.to(self.device)

        # Wrap the network for multiple GPUs if available.
        if torch.cuda.device_count() > 1:
            print("Using DataParallel on {} GPUs".format(torch.cuda.device_count()))
            self.net_G = torch.nn.DataParallel(self.net_G, device_ids=args.gpu_ids)

        # Learning rate
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr, weight_decay=0)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
                                           betas=(0.9, 0.999), weight_decay=0.1)

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        # training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self.shuffle_AB = args.shuffle_AB

        # define the loss functions
        self.multi_scale_train = args.multi_scale_train
        self.multi_scale_infer = args.multi_scale_infer
        self.weights = tuple(args.multi_pred_weights)

        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'fl':
            print('\n Calculating alpha in Focal-Loss (FL) ...')
            alpha = get_alpha(dataloaders['train']) # class frequencies
            print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
            self._pxl_loss = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2, smooth = 1e-5)
        elif args.loss == "miou":
            print('\n Calculating Class occurrences in training set...')
            alpha = np.asarray(get_alpha(dataloaders['train']))
            alpha = alpha / np.sum(alpha)
            weights = 1 - torch.from_numpy(alpha).cuda()
            print(f"Weights = {weights}")
            self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=args.n_class).cuda()
        elif args.loss == "mmiou":
            self._pxl_loss = mmIoULoss(n_classes=args.n_class).cuda()
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # Create checkpoint & visualization dirs if not exist
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists(self.vis_dir):
            os.mkdir(self.vis_dir)

        # ----------------------------
        # NEW: Lists to track metrics
        # ----------------------------
        self.training_losses = []
        self.validation_losses = []
        self.gradient_norms = []   # track per iteration
        # For calibration curve (we’ll gather these on the final val pass)
        self.final_val_probs = []
        self.final_val_labels = []


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        print("\n")
        if self.args.pretrain is not None and os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        elif self.args.pretrain is not None:
            print("Initializing backbone weights from: " + self.args.pretrain)
            self.net_G.load_state_dict(torch.load(self.args.pretrain), strict=False)
            self.net_G.to(self.device)
            self.net_G.eval()
        else:
            print('training from scratch...')
        print("\n")

    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id
        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_final_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_final_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()

        m = len(self.dataloaders['train']) if self.is_training else len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = ('Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %
                       (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                        imps*self.batch_size, est, self.G_loss.item(), running_acc))
            self.logger.write(message)

        if np.mod(self.batch_id, 500) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
            vis_pred = utils.make_numpy_grid(self._visualize_pred())
            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                                     str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
                          (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n\n')

    def _update_checkpoints(self):
        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Latest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
                          % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)

        if self.multi_scale_infer == "True":
            self.G_final_pred = torch.zeros(self.G_pred[-1].size()).to(self.device)
            for pred in self.G_pred:
                if pred.size(2) != self.G_pred[-1].size(2):
                    self.G_final_pred += F.interpolate(pred, size=self.G_pred[-1].size(2), mode="nearest")
                else:
                    self.G_final_pred += pred
            self.G_final_pred = self.G_final_pred / len(self.G_pred)
        else:
            self.G_final_pred = self.G_pred[-1]

    def _backward_G(self):
        gt = self.batch['L'].to(self.device).float()
        if self.multi_scale_train == "True":
            temp_loss = 0.0
            for i, pred in enumerate(self.G_pred):
                # Resize ground truth if needed.
                if pred.size(2) != gt.size(2):
                    gt_resized = F.interpolate(gt, size=pred.size()[2:], mode="nearest")
                else:
                    gt_resized = gt

                # Create a valid mask: valid pixels are those where gt != 2.
                valid_mask = (gt_resized != 2).float()
                
                # Compute pixel loss without reduction (i.e., per-pixel loss).
                # Make sure your self._pxl_loss supports reduction='none'
                loss_per_pixel = self._pxl_loss(pred, gt_resized, reduction='none')
                
                # Debug prints (optional, remove in production)
                # print(f"Scale {i}: loss_per_pixel shape: {loss_per_pixel.shape}, valid_mask unique values: {valid_mask.unique()}")

                # Compute masked loss: sum over valid pixels and normalize.
                scale_loss = self.weights[i] * (loss_per_pixel * valid_mask).sum() / (valid_mask.sum() + 1e-6)
                temp_loss += scale_loss

            self.G_loss = temp_loss
        else:
            # For single scale training, assume you want to ignore gt==2 similarly.
            gt_resized = gt
            valid_mask = (gt_resized != 2).float()
            loss_per_pixel = self._pxl_loss(self.G_pred[-1], gt_resized, reduction='none')
            self.G_loss = (loss_per_pixel * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        self.G_loss.backward()



    def train_models(self):

        self._load_checkpoint()

        patience = 10
        no_improvement_count = 0
        best_val_loss = float('inf')

        # Loop over epochs
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            # --------------------
            # TRAIN
            # --------------------
            self._clear_cache()
            self.is_training = True
            self.net_G.train()

            # Track total train loss this epoch
            train_loss = 0.0
            train_samples = 0

            total = len(self.dataloaders['train'])
            self.logger.write('lr: %0.7f\n \n' % self.optimizer_G.param_groups[0]['lr'])

            for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train'], 0), total=total):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()

                # -------------
                # GRADIENT NORM
                # -------------
                grad_norm = 0.0
                for p in self.net_G.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item()**2
                grad_norm = grad_norm**0.5
                self.gradient_norms.append(grad_norm)

                self.optimizer_G.step()

                # track total training loss
                bs = batch['L'].size(0)
                train_loss += self.G_loss.item() * bs
                train_samples += bs

                # log
                self._collect_running_batch_states()
                self._timer_update()

            # Average train loss
            train_loss /= max(train_samples, 1)
            self.training_losses.append(train_loss)

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            # --------------------
            # VALIDATION
            # --------------------
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            val_loss = 0.0
            val_samples = 0

            # Reset the lists for calibration curve data.
            self.final_val_probs = []
            self.final_val_labels = []
            val_loader = self.dataloaders['val']
            print(f"Validation set size: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)

                    # Compute validation loss with masking.
                    gt = batch['L'].to(self.device).float()
                    valid_mask = (gt != 2).float()
                    loss_val = self._pxl_loss(self.G_pred[-1], gt, reduction='none')
                    loss_val = (loss_val * valid_mask).sum() / (valid_mask.sum() + 1e-6)
                    bs = gt.size(0)
                    print(f"  batch {self.batch_id}:   bs={bs}   valid_pixels={(gt != 2).sum().item()}   raw_loss={loss_val.item():.6f}")

                    val_loss += loss_val.item() * bs
                    val_samples += bs

                    # For calibration curve, store predicted probabilities and labels only for valid pixels.
                    probs = torch.softmax(self.G_final_pred, dim=1)[:, 1].cpu().numpy().flatten()
                    labels = batch['L'].cpu().numpy().flatten()
                    valid_indices = (labels != 2)
                    self.final_val_probs.extend(probs[valid_indices])
                    self.final_val_labels.extend(labels[valid_indices])

                self._collect_running_batch_states()

            val_loss /= max(val_samples, 1)
            self.validation_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count  +=1
            
            if no_improvement_count >= patience:
                self.logger.write(f"Validation loss did not improve for {patience} epochs. Stopping early.\n")
                break


            self._collect_epoch_states()
            self._update_val_acc_curve()
            self._update_checkpoints()

        # ------------------------------------------
        # Done training: Plot all desired curves now
        # ------------------------------------------

        # 1) LOSS CURVE (train vs. val)
        plt.figure()
        plt.plot(self.training_losses, label='Train Loss')
        plt.plot(self.validation_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.savefig(os.path.join(self.checkpoint_dir, 'loss_curve.png'))
        plt.close()

        # 2) GRADIENT NORM CURVE
        plt.figure()
        plt.plot(self.gradient_norms)
        plt.xlabel('Training Iteration')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms Over Training')
        plt.savefig(os.path.join(self.checkpoint_dir, 'gradient_norms.png'))
        plt.close()

        # 3) CALIBRATION CURVE (final epoch’s validation)
        #    We use the final_val_probs and final_val_labels
        prob_true, prob_pred = calibration_curve(self.final_val_labels, self.final_val_probs, n_bins=10)
        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        # Perfectly calibrated line:
        plt.plot([0,1],[0,1], linestyle='--', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve (Final Val)')
        plt.legend()
        plt.savefig(os.path.join(self.checkpoint_dir, 'calibration_curve.png'))
        plt.close()

        self.logger.write("Training complete. Curves saved to %s\n" % self.checkpoint_dir)
