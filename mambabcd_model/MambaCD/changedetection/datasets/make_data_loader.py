import argparse
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import MambaCD.changedetection.datasets.imutils as imutils


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
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.crop_size = crop_size

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

        # Build a mapping from global index to (orig_index, aug_idx, is_deforested)
        # This way, each chip that is deforested gets 3 entries; non-deforested gets 1.
        self.index_mapping = []
        for i, filename in enumerate(self.data_list):
            # Build label path
            label_path = os.path.join(self.dataset_path, 'GT', filename)
            if not label_path.endswith('.png'):
                label_path += '.png'
            try:
                label = self.loader(label_path)
            except Exception as e:
                print(f"Error loading label for {filename}: {e}")
                continue
            label = label / 255.0  # Normalize label

            is_deforested = np.any(label > 0)
            if 'train' in self.data_pro_type:
                #setting to false to not augment data
                #is_deforested = False
                if is_deforested:
                    # Add 3 augmentations for deforested chips.
                    for aug_idx in range(3):
                        self.index_mapping.append((i, aug_idx, True))
                else:
                    # Only one version for non-deforested chips.
                    self.index_mapping.append((i, 0, False))
            else:
                # For validation/testing, no augmentation: always aug_idx 0.
                self.index_mapping.append((i, 0, False))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, index):
        # Retrieve the mapping for this global index.
        orig_index, aug_idx, is_deforested = self.index_mapping[index]
        #set to 0 so i dont augment, comment out for augmentation
        #aug_idx= 0
        # Build file paths
        pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[orig_index])
        post_path = os.path.join(self.dataset_path, 'T2', self.data_list[orig_index])
        label_path = os.path.join(self.dataset_path, 'GT', self.data_list[orig_index])

        if not pre_path.endswith('.png'):
            pre_path += '.png'
        if not post_path.endswith('.png'):
            post_path += '.png'
        if not label_path.endswith('.png'):
            label_path += '.png'

        # Load images and label
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255.0  # Normalize label

        # Decide which augmentation pipeline to use
        if 'train' in self.data_pro_type:
            #setting to false to not augment data
            #is_deforested = False
            if is_deforested:
                augmented_samples = self.__transforms(True, pre_img, post_img, label)
            else:
                augmented_samples = self.__transforms(False, pre_img, post_img, label)
        else:
            augmented_samples = self.__transforms(False, pre_img, post_img, label)

        # For deforested chips, augmented_samples should have 3 entries.
        # For non-deforested chips, it has 1 entry, and aug_idx will always be 0.
        pre_img_aug, post_img_aug, label_aug = augmented_samples[aug_idx]
        data_idx = self.data_list[orig_index]
        return pre_img_aug, post_img_aug, label_aug, data_idx


    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            augmented_samples = []
            for angle in [0, 90, 180]:
                # Make copies to avoid in-place modifications.
                pre_img_copy = pre_img.copy()
                post_img_copy = post_img.copy()
                label_copy = label.copy()
                
                # Apply deterministic rotation if needed.
                if angle == 90:
                    pre_img_copy = np.rot90(pre_img_copy, k=1, axes=(0, 1))
                    post_img_copy = np.rot90(post_img_copy, k=1, axes=(0, 1))
                    label_copy = np.rot90(label_copy, k=1)
                elif angle == 180:
                    pre_img_copy = np.rot90(pre_img_copy, k=2, axes=(0, 1))
                    post_img_copy = np.rot90(post_img_copy, k=2, axes=(0, 1))
                    label_copy = np.rot90(label_copy, k=2)
                pre_img_copy = np.ascontiguousarray(pre_img_copy)
                post_img_copy = np.ascontiguousarray(post_img_copy)
                label_copy = np.ascontiguousarray(label_copy)
                
                # Apply random augmentations.
                pre_img_copy, post_img_copy, label_copy = imutils.random_crop_new(
                    pre_img_copy, post_img_copy, label_copy, self.crop_size)
                pre_img_copy, post_img_copy, label_copy = imutils.random_fliplr(
                    pre_img_copy, post_img_copy, label_copy)
                pre_img_copy, post_img_copy, label_copy = imutils.random_flipud(
                    pre_img_copy, post_img_copy, label_copy)
                pre_img_copy, post_img_copy, label_copy = imutils.random_rot(
                    pre_img_copy, post_img_copy, label_copy)
                
                # Standardization.
                global_mean_rgb = np.array([338.86, 540.61, 453.62])
                global_std_rgb  = np.array([149.73, 199.71, 298.44])
                pre_img_copy[..., :3] = (pre_img_copy[..., :3] - global_mean_rgb) / global_std_rgb
                post_img_copy[..., :3] = (post_img_copy[..., :3] - global_mean_rgb) / global_std_rgb
                
                # Transpose from HWC to CHW.
                pre_img_copy = np.transpose(pre_img_copy, (2, 0, 1))
                post_img_copy = np.transpose(post_img_copy, (2, 0, 1))
                
                augmented_samples.append((pre_img_copy, post_img_copy, label_copy))
            
            # Return a list of 3 augmented samples.
            return augmented_samples
        else:
            # No augmentation: return a single sample in a list.
            global_mean_rgb = np.array([338.86, 540.61, 453.62])
            global_std_rgb  = np.array([149.73, 199.71, 298.44])
            pre_img[..., :3] = (pre_img[..., :3] - global_mean_rgb) / global_std_rgb
            post_img[..., :3] = (post_img[..., :3] - global_mean_rgb) / global_std_rgb
            pre_img = np.transpose(pre_img, (2, 0, 1))
            post_img = np.transpose(post_img, (2, 0, 1))
            return [(pre_img, post_img, label)]

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


def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'DEFORESTATION' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU' in args.dataset:
        dataset = ChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=1,
                                 drop_last=False)
        return data_loader
    elif 'xBD' in args.dataset:
        dataset = DamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=6,
                                 drop_last=False)
        return data_loader
    
    elif 'SECOND' in args.dataset:
        dataset = SemanticChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='WHUBCD')
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD')
    parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

    with open(args.data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())