import sys
sys.path.append('/home/songjian/project/MambaCD')

import argparse
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.MambaBCD import STMambaBCD
import imageio
import MambaCD.changedetection.utils_func.lovasz_loss as L


class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        # ================
        # 1) Setup model, evaluator, and paths
        # ================
        self.evaluator = Evaluator(num_class=2)
        self.deep_model = STMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
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
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        ).cuda()

        self.epoch = args.max_iters // args.batch_size
        
        # Directory to save final binary change maps
        self.change_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'change_map')
        os.makedirs(self.change_map_saved_path, exist_ok=True)

        # Directory to save calibration curve & sample outputs
        self.model_save_path = os.path.join(args.result_saved_path, args.dataset, args.model_type)
        os.makedirs(self.model_save_path, exist_ok=True)

        # Resume checkpoint if provided
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
        
        # ================
        # 2) Create Validation & Test DataLoaders
        # ================
        # Validation set
        self.val_dataset = ChangeDetectionDatset(
            self.args.val_dataset_path,
            self.args.val_data_name_list,
            self.args.crop_size,
            None,
            'val'
        )
        self.val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=1,
            drop_last=False,
            persistent_workers=True
        )

        # Test set
        self.test_dataset = ChangeDetectionDatset(
            self.args.test_dataset_path,
            self.args.test_data_name_list,
            self.args.crop_size,
            None,
            'test'
        )
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=1,
            drop_last=False
        )

        self.deep_model.eval()

    def infer(self):
        """
        Runs inference on the test set to:
        1) Collect probabilities for the "changed" class.
        2) Determine the best threshold via F0.5 analysis.
        3) Generate and save composite images arranged as:
            Top: Before image,
            Middle: [Actual mask | Predicted mask],
            Bottom: After image.
        4) Evaluate final metrics.
        """
        torch.cuda.empty_cache()
        self.evaluator.reset()
        
        # Lists to store probabilities, labels, names, and the before/after images.
        all_probs = []
        all_labels = []
        all_names = []
        all_pre_imgs = []
        all_post_imgs = []
        
        with torch.no_grad():
            for itera, data in enumerate(self.test_data_loader):
                #print("Processing chip number:", itera + 1)
                pre_change_imgs, post_change_imgs, labels, names = data
                #print("Current chip name:", names[0])
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                # Store the before and after images (on CPU) for later visualization.
                all_pre_imgs.append(pre_change_imgs.cpu().numpy())
                all_post_imgs.append(post_change_imgs.cpu().numpy())

                # Output probabilities (with inference=True so softmax is applied)
                output = self.deep_model(pre_change_imgs, post_change_imgs, inference=True)
                #print("Output shape:", output.shape)  # Expected: [1, 2, H, W]
                
                # Extract probability map for the "changed" class.
                prob_map = output[:, 1, :, :] if output.shape[1] > 1 else output[:, 0, :, :]
                #print("Prob_map shape before squeeze:", prob_map.shape)  # Expected: [1, H, W]
                
                all_probs.append(prob_map.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_names.append(names)
        
        # Determine F0.5-based threshold from the collected probabilities.
        flat_probs = np.concatenate([p.flatten() for p in all_probs])
        flat_labels = np.concatenate([l.flatten() for l in all_labels])
        
        from sklearn.metrics import precision_recall_curve
        precision, recall, pr_thresholds = precision_recall_curve(flat_labels, flat_probs)
        beta = 0.5
        f05_scores = (1 + beta**2) * (precision[:-1] * recall[:-1]) / (beta**2 * precision[:-1] + recall[:-1] + 1e-8)
        optimal_idx = np.argmax(f05_scores)
        optimal_threshold = pr_thresholds[optimal_idx]
        print("Optimal threshold from Precision-Recall analysis (F0.5):", optimal_threshold)
        
        # 3) Generate composite images for each chip.
        # Global normalization parameters (as used during training)
        global_mean_rgb = np.array([338.86, 540.61, 453.62])
        global_std_rgb  = np.array([149.73, 199.71, 298.44])
        for idx, (prob_map, lbl, names, pre_img, post_img) in enumerate(
                zip(all_probs, all_labels, all_names, all_pre_imgs, all_post_imgs)):
            # Set output filename.
            image_name = names[0] + '.png'
            
            # Predicted mask: threshold the probability map.
            pred_mask = (np.squeeze(prob_map) > optimal_threshold).astype(np.uint8) * 255
            
            # Actual (ground truth) mask.
            gt_mask = np.squeeze(lbl)
            if gt_mask.max() <= 1:
                gt_mask = (gt_mask * 255).astype(np.uint8)
            else:
                gt_mask = gt_mask.astype(np.uint8)
            
            # Process before (pre-change) image.
            # Assuming pre_img has shape [1, C, H, W]; squeeze to [C, H, W]
            pre_img = np.squeeze(pre_img)
            # If color (3 channels), convert from channel-first to channel-last.
            if pre_img.ndim == 3:
                pre_img = np.transpose(pre_img, (1, 2, 0))
                for c in range(3):
                    pre_img[..., c] = pre_img[..., c] * global_std_rgb[c] + global_mean_rgb[c]
                # Optionally convert to uint8 if needed
                pre_img = np.clip(pre_img, 0, 255)
                pre_img = (pre_img * 255).astype(np.uint8) if pre_img.max() <= 1 else pre_img.astype(np.uint8)
            else:
                pre_img = pre_img.astype(np.uint8)
            
            # Process after (post-change) image similarly.
            post_img = np.squeeze(post_img)
            if post_img.ndim == 3:
                post_img = np.transpose(post_img, (1, 2, 0))
                for c in range(3):
                    post_img[..., c] = post_img[..., c] * global_std_rgb[c] + global_mean_rgb[c]
                post_img = np.clip(post_img, 0, 255)
                post_img = (post_img * 255).astype(np.uint8) if post_img.max() <= 1 else post_img.astype(np.uint8)
            else:
                post_img = post_img.astype(np.uint8)
            
            # Create a composite image with 3 rows:
            # Row 1: Before image (spanning full width)
            # Row 2: Two images side by side (left: actual mask, right: predicted mask)
            # Row 3: After image (spanning full width)
            # For simplicity, we'll use matplotlib subplots to arrange these and then save the figure.

            # Create a figure with GridSpec: 3 rows and 2 columns.
            fig = plt.figure(figsize=(10, 12))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

            # Row 1: Before image (spanning both columns)
            ax0 = fig.add_subplot(gs[0, :])
            ax0.imshow(pre_img)
            ax0.set_title("Before Image")
            ax0.axis('off')

            # Row 2, Column 1: Actual mask
            ax1 = fig.add_subplot(gs[1, 0])
            ax1.imshow(gt_mask, cmap='gray')
            ax1.set_title("Actual Mask")
            ax1.axis('off')

            # Row 2, Column 2: Predicted mask
            ax2 = fig.add_subplot(gs[1, 1])
            ax2.imshow(pred_mask, cmap='gray')
            ax2.set_title("Predicted Mask")
            ax2.axis('off')

            # Row 3: After image (spanning both columns)
            ax3 = fig.add_subplot(gs[2, :])
            ax3.imshow(post_img)
            ax3.set_title("After Image")
            ax3.axis('off')

            plt.tight_layout()
            composite_path = os.path.join(self.change_map_saved_path, image_name)
            plt.savefig(composite_path)
            plt.close(fig)

        
        # 4) Evaluate final metrics on the thresholded predictions.
        self.evaluator.reset()
        for lbl, prob in zip(all_labels, all_probs):
            binary_pred = (np.squeeze(prob) > optimal_threshold).astype(np.uint8)
            if lbl.ndim == 3 and binary_pred.ndim == 2:
                binary_pred = np.expand_dims(binary_pred, axis=0)
            self.evaluator.add_batch(lbl, binary_pred)
        
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        
        print(f'Final Evaluation - Recall: {rec}, Precision: {pre}, '
            f'OA: {oa}, F1: {f1_score}, IoU: {iou}, Kappa: {kc}')
        print('Inference stage with optimal threshold is done!')


def main():
    parser = argparse.ArgumentParser(description="Inference on Deforestation dataset")
    parser.add_argument('--cfg', type=str,
                        default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='DEFORESTATION')

    # Paths
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--val_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--val_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--result_saved_path', type=str, default='../results')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD_Small')
    parser.add_argument('--resume', type=str)

    args = parser.parse_args()

    # Read test data list
    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    # Read val data list
    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [data_name.strip() for data_name in f]
    args.val_data_name_list = val_data_name_list

    # Instantiate inference class & run
    infer = Inference(args)
    infer.infer()

    # Optionally run final validation analysis (calibration curve, again threshold with F0.5, etc.)


if __name__ == "__main__":
    main()
