import argparse
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import MambaCD.changedetection.datasets.imutils as imutils
import random




def npz_loader(path):
    """Loads data from an NPZ file. Expects the array to be stored under key 'array'."""
    if path.endswith('.npz'):
        data = np.load(path)
        img = data['array']
    else:
        img = np.load(path)
    # If the image is channels-first, convert it to channels-last
    if img.ndim == 3 and img.shape[0] == 4:  # Assuming 4 bands means first dim is channels
        img = np.transpose(img, (1, 2, 0))
    return img

    
def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot

class ChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, type, data_loader=npz_loader):
        """
        Args:
            dataset_path: Root directory containing folders 'before', 'after', and 'mask'.
            data_list: List of chip filenames (without extensions).
            type: 'train' or 'test'/'val'; used to determine augmentation.
            data_loader: Function used to load files. Defaults to npz_loader.
        """
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type


    def __transforms(self, aug, pre_img, post_img, label):
        """Apply augmentations if required, then normalize and transpose images."""
        if aug:
            #pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)
        pre_img = imutils.normalize_img(pre_img)
        pre_img = np.transpose(pre_img, (2, 0, 1))
        post_img = imutils.normalize_img(post_img)
        post_img = np.transpose(post_img, (2, 0, 1))
        return pre_img, post_img, label

    def __getitem__(self, index):
        # Get the base file name and add the "chip_" prefix if needed.
        filename = self.data_list[index]
        if not filename.startswith("chip_"):
            filename = f"chip_{filename}"

        # Build full paths and append the .npz extension if missing.
        pre_path = os.path.join(self.dataset_path, 'before', filename)
        post_path = os.path.join(self.dataset_path, 'after', filename)
        label_path = os.path.join(self.dataset_path, 'mask', filename)
        if not pre_path.endswith('.npz'):
            pre_path += '.npz'
        if not post_path.endswith('.npz'):
            post_path += '.npz'
        if not label_path.endswith('.npz'):
            label_path += '.npz'

        # Load the before, after, and mask images.
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
     
        # Determine if the chip is deforested.
        is_deforested = np.any(label == 1)

        # Apply augmentation if in training mode.
        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(is_deforested, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)
    











# ── utility ─────────────────────────────────────────────────────────────────────

def _deforestation_ratio(mask: np.ndarray) -> float:
    """Fraction of pixels with value == 1."""
    return (mask == 1).mean()

# ── balanced dataset ────────────────────────────────────────────────────────────

class BalancedChangeDetectionDataset(ChangeDetectionDatset):
    """
    Like ChangeDetectionDatset but returns an *equal* number of deforested and
    non‑deforested chips every epoch.

    A chip is *positive* if ≥ `thr` fraction of its pixels are 1 in the mask.
    """

    def __init__(self, dataset_path, data_list, split,
                 thr: float = 0.02, data_loader=npz_loader):
        super().__init__(dataset_path, data_list, split, data_loader)

        self.thr = thr
        self.pos_idx, self.neg_idx = self._build_index_lists()
        if not self.pos_idx:
            raise RuntimeError(f"No chips with ≥{thr*100:.1f}% deforestation!")

        # For reproducible val runs:
        self.rng = random.Random(0 if 'val' in split else None)

    # --------------------------------------------------------------------------

    def _build_index_lists(self):
        pos, neg = [], []
        for i, fname in enumerate(self.data_list):
            # build mask path once
            f = fname if fname.startswith("chip_") else f"chip_{fname}"
            if not f.endswith(".npz"):
                f += ".npz"
            mask_path = os.path.join(self.dataset_path, "mask", f)

            # load mask & classify
            mask = self.loader(mask_path)
            (_deforestation_ratio(mask) >= self.thr and pos or neg).append(i)
        return pos, neg

    # --------------------------------------------------------------------------

    def __len__(self):
        # show the DataLoader a balanced view
        return 2 * len(self.pos_idx)

    # --------------------------------------------------------------------------

    def __getitem__(self, index):
        # even → positive, odd → random negative
        if index % 2 == 0:                         # positive
            real_idx = self.pos_idx[index // 2]
        else:                                     # negative (fresh draw each call)
            real_idx = self.rng.choice(self.neg_idx)
        return super().__getitem__(real_idx)      # defer to parent implementation








class SemanticChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(pre_img, post_img, cd_label, t1_label, t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label, t1_label, t2_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index] + '.png')
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index] + '.png')
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index] + '.png')
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index] + '.png')
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index] + '.png')
        else:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(T1_label_path)
        t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label, t2_label)
        else:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(False, pre_img, post_img, cd_label, t1_label, t2_label)
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label, t1_label, t2_label, data_idx

    def __len__(self):
        return len(self.data_list)


class DamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label, self.crop_size)
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type: 
            parts = self.data_list[index].rsplit('_', 2)

            pre_img_name = f"{parts[0]}_pre_disaster_{parts[1]}_{parts[2]}.png"
            post_img_name = f"{parts[0]}_post_disaster_{parts[1]}_{parts[2]}.png"

            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            
            loc_label_path = os.path.join(self.dataset_path, 'masks', pre_img_name)
            clf_label_path = os.path.join(self.dataset_path, 'masks', post_img_name)
        else:
            pre_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_pre_disaster.png')
            post_path = os.path.join(self.dataset_path, 'images', self.data_list[index] + '_post_disaster.png')
            loc_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_pre_disaster.png')
            clf_label_path = os.path.join(self.dataset_path, 'masks', self.data_list[index]+ '_post_disaster.png')

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        loc_label = self.loader(loc_label_path)[:,:,0]
        clf_label = self.loader(clf_label_path)[:,:,0]

        if 'train' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
            clf_label[clf_label == 0] = 255
        else:
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)


# def make_data_loader(args, **kwargs):  # **kwargs could be omitted
#     if 'DEFORESTATION' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU' in args.dataset:
#         dataset = ChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.type)
#         # train_sampler = DistributedSampler(dataset, shuffle=True)
#         data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=1,
#                                  drop_last=False)
#         return data_loader
#     elif 'xBD' in args.dataset:
#         dataset = DamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
#         data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=6,
#                                  drop_last=False)
#         return data_loader
    
#     elif 'SECOND' in args.dataset:
#         dataset = SemanticChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
#         data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
#                                  drop_last=False)
#         return data_loader
    
#     else:
#         raise NotImplementedError

def make_data_loader(args, phase: str = "train", **kwargs):
    """
    Factory that returns a DataLoader for the requested phase.

    For 'train' and 'val' it balances the stream on‑the‑fly:
        – chips whose mask has >= deforestation_thr pixels == 1
        – an equal number of randomly drawn chips with <  deforestation_thr
    For 'test' it returns the full, unbiased dataset.

    """
    if "DEFORESTATION" not in args.dataset.upper():
        raise ValueError("make_data_loader currently supports only *DEFORESTATION* datasets")

    # --------------------------------------------------------------------- paths
    if phase == "train":
        dataset_path   = args.train_dataset_path
        data_name_list = args.train_data_name_list
        shuffle        = True
        dataset_cls    = BalancedChangeDetectionDataset
    elif phase == "val":
        dataset_path   = args.val_dataset_path
        data_name_list = args.val_data_name_list
        shuffle        = False
        dataset_cls    = BalancedChangeDetectionDataset
    elif phase == "test":
        dataset_path   = args.test_dataset_path
        data_name_list = args.test_data_name_list
        shuffle        = False
        dataset_cls    = ChangeDetectionDatset           # full set
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    # ------------------------------------------------------------------- dataset
    dataset = dataset_cls(
        dataset_path,
        data_name_list,
        phase,
        thr=getattr(args, "deforestation_thr", 0.02)      # 2 % threshold by default
    )

    # --------------------------------------------------------------- data loader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=True,
        drop_last=False,
        **kwargs,
    )
    return loader




# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
#     parser.add_argument('--dataset', type=str, default='WHUBCD')
#     parser.add_argument('--max_iters', type=int, default=10000)
#     parser.add_argument('--type', type=str, default='train')
#     parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD')
#     parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt')
#     parser.add_argument('--shuffle', type=bool, default=True)
#     parser.add_argument('--batch_size', type=int, default=8)
#     parser.add_argument('--data_name_list', type=list)

#     args = parser.parse_args()

#     with open(args.data_list_path, "r") as f:
#         # data_name_list = f.read()
#         data_name_list = [data_name.strip() for data_name in f]
#     args.data_name_list = data_name_list
#     train_data_loader = make_data_loader(args)
#     for i, data in enumerate(train_data_loader):
#         pre_img, post_img, labels, _ = data
#         pre_data, post_data = Variable(pre_img), Variable(post_img)
#         labels = Variable(labels)
#         print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())