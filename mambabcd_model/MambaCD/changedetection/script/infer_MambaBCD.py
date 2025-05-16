import sys 
sys.path.append('/home/songjian/project/MambaCD')

import argparse
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import f1_score

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
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import precision_recall_curve, f1_score as sk_f1_score

class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

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
            mlp_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
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
        if args.resume:
            sd = torch.load(args.resume, map_location='cpu')
            # strip DataParallel “module.” prefix if present
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            # load without strict checking
            self.deep_model.load_state_dict(sd, strict=False)
            print(f"Loaded {len(sd)} tensors from {args.resume}")
                
        # ================ 2) Create Test DataLoaders ================
        self.test_data_loader = make_data_loader(args, phase='test')

        self.deep_model.eval()

    def infer(self):
        """
        Full-test inference:
        1. Run model on every test chip.
        2. Find optimal F-0.5 threshold (ignoring label==2).
        3. Compute pixel-level metrics.
        4. Compute per-chip F1 scores and select N worst.
        5. Produce ONE composite PNG (4 rows × N columns) for the worst chips.
        """


        torch.cuda.empty_cache()
        self.evaluator.reset()

        # ------------------------------------------------------------------ #
        # 1) Inference pass – collect probability maps, labels & names
        # ------------------------------------------------------------------ #
        all_probs, all_labels, all_names = [], [], []
        all_pre_imgs, all_post_imgs = [], []

        with torch.no_grad():
            for pre, post, lbl, names in self.test_data_loader:
                pre  = pre.cuda().float()
                post = post.cuda().float()
                lbl  = lbl.cuda().long()

                logits = self.deep_model(pre, post, inference=True)
                prob   = logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]

                all_probs.extend(prob.cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())
                all_names.extend(names)
                all_pre_imgs.extend(pre.cpu().numpy())
                all_post_imgs.extend(post.cpu().numpy())

        # ------------------------------------------------------------------ #
        # 2) Find optimal F-0.5 threshold (ignore label==2)
        # ------------------------------------------------------------------ #
        flat_p, flat_y = [], []
        for p, y in zip(all_probs, all_labels):
            valid = (y != 2)
            flat_p.append(p[valid].ravel())
            flat_y.append((y[valid] == 1).ravel())
        flat_p = np.concatenate(flat_p)
        flat_y = np.concatenate(flat_y)

        pr, rc, th = precision_recall_curve(flat_y, flat_p)
        beta = 0.5
        f05  = (1 + beta**2) * pr[:-1] * rc[:-1] / (beta**2 * pr[:-1] + rc[:-1] + 1e-8)
        optimal_thr = th[np.argmax(f05)]
        print(f"Optimal F-0.5 threshold: {optimal_thr:.4f}")

        # ------------------------------------------------------------------ #
        # 3) Compute per-chip F1 scores
        # ------------------------------------------------------------------ #
        chip_f1s = []
        for probs, labels, names in zip(all_probs, all_labels, all_names):
            valid = (labels != 2)
            y_true = (labels[valid] == 1).ravel()
            y_pred = (probs[valid] > optimal_thr).ravel()
            f1 = sk_f1_score(y_true, y_pred, zero_division=1)

            # normalize name to str
            name = names[0] if isinstance(names, (list, tuple)) else names
            chip_f1s.append((name, f1))

        # sort ascending and report
        chip_f1s.sort(key=lambda x: x[1])
        print("\nLowest-F1 chips:")
        for name, score in chip_f1s[:5]:
            print(f"  {name:10s} : F1 = {score:.4f}")

        # ------------------------------------------------------------------ #
        # 4) Compute global pixel-level metrics at 0.5
        # ------------------------------------------------------------------ #
        for y, p in zip(all_labels, all_probs):
            m = (y != 2)
            self.evaluator.add_batch(
                ((y[m] == 1).astype(np.uint8)),
                ((p[m] > 0.5).astype(np.uint8))
            )

        print("\nGlobal pixel metrics (threshold=0.5):")
        print("Confusion matrix:\n", self.evaluator.confusion_matrix)
        print(f"Recall   : {self.evaluator.Pixel_Recall_Rate():.4f}")
        print(f"Precision: {self.evaluator.Pixel_Precision_Rate():.4f}")
        print(f"F1       : {self.evaluator.Pixel_F1_score():.4f}")
        print(f"F0.5     : {self.evaluator.Pixel_F05_score():.4f}")
        print(f"IoU      : {self.evaluator.Intersection_over_Union():.4f}")
        print(f"Kappa    : {self.evaluator.Kappa_coefficient():.4f}")

        eval_opt = Evaluator(num_class=2)
        for y, p in zip(all_labels, all_probs):
            m = (y != 2)
            eval_opt.add_batch(
                (y[m] == 1).astype(np.uint8),
                (p[m] > optimal_thr).astype(np.uint8)
            )

        print(f"Global pixel metrics (threshold = {optimal_thr:.4f}):")
        print("  Confusion matrix:\n", eval_opt.confusion_matrix)
        print(f"  Recall   : {eval_opt.Pixel_Recall_Rate():.4f}")
        print(f"  Precision: {eval_opt.Pixel_Precision_Rate():.4f}")
        print(f"  F1       : {eval_opt.Pixel_F1_score():.4f}")
        print(f"  F0.5     : {eval_opt.Pixel_F05_score():.4f}")
        print(f"  IoU      : {eval_opt.Intersection_over_Union():.4f}")
        print(f"  Kappa    : {eval_opt.Kappa_coefficient():.4f}\n")

        # ------------------------------------------------------------------ #
        # 5) Build composite for N worst-F1 chips
        # ------------------------------------------------------------------ #
        N = 4
        worst_names = [name for name, _ in chip_f1s[N:2*N]]

        worst_chips = []
        for p, y, pre, post, names in zip(all_probs, all_labels,
                                        all_pre_imgs, all_post_imgs, all_names):
            name = names[0] if isinstance(names, (list, tuple)) else names
            if name in worst_names:
                worst_chips.append((name, p, y, pre, post))

        # preserve F1 order
        worst_chips.sort(key=lambda x: worst_names.index(x[0]))

        if not worst_chips:
            print("Warning: no chips matched the worst-F1 list!")
            return

        # helper for false-color visualization
        def false_color(tensor):
            arr = tensor if isinstance(tensor, np.ndarray) else tensor.cpu().numpy()
            b2, b3, b8 = arr[0], arr[1], arr[3]
            B = b2 * 602.05 + 632.43
            G = b3 * 570.71 + 796.16
            R = b8 * 592.58 + 2918.27

            def stretch(c):
                vmin, vmax = np.percentile(c, [2, 98])
                if vmax - vmin < 1e-6:
                    return np.zeros_like(c, dtype=np.uint8)
                return ((np.clip(c, vmin, vmax) - vmin) / (vmax - vmin) * 255).astype(np.uint8)

            return np.stack([stretch(R), stretch(G), stretch(B)], axis=-1)

        # plot 4 rows × N columns
        ncols = len(worst_chips)
        fig = plt.figure(figsize=(3 * ncols, 9))
        gs  = gridspec.GridSpec(4, ncols, wspace=0.005, hspace=0.001)
        plt.subplots_adjust(left=0.30, wspace=0.005, hspace=0.001)

        row_labels = ["T₁ 2020", "T₂ 2021", "Prediction", "Ground truth"]
        for col, (chip_name, probs, labels, pre, post) in enumerate(worst_chips):
            imgs = [
                false_color(pre.squeeze()),
                false_color(post.squeeze()),
                (probs.squeeze() > optimal_thr).astype(np.uint8) * 255,
                (labels == 1).astype(np.uint8) * 255
            ]
            cmaps = [None, None, "gray", "gray"]

            for row in range(4):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(imgs[row], cmap=cmaps[row])
                ax.axis("off")

                if col == 0:
                    ax.text(
                        x = -0.05,
                        y = 0.5,
                        s = row_labels[row],
                        transform = ax.transAxes,
                        fontsize = 12,
                        va = "center",
                        ha = "right"
                    )

        out_path = os.path.join(self.change_map_saved_path, f"worst_{N}_f1_chips.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"Saved composite to {out_path}")

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
    parser.add_argument('--result_saved_path', type=str, default='../results')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD_Tiny')
    parser.add_argument('--resume', type=str, default='../../../saved_models/DEFORESTATION/MambaBCD_Small_1744601731.8273776/best_model_epoch7.pth')

    args = parser.parse_args()

    # Read test data list
    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list


    # Instantiate inference class & run
    infer = Inference(args)
    infer.infer()

    # Optionally run final validation analysis (calibration curve, again threshold with F0.5, etc.)


if __name__ == "__main__":
    main()
