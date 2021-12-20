import os
import cv2
import glob
import numpy as np
import torch
from matplotlib import pyplot as plt

# from modules
from torch.utils.data import Dataset, DataLoader
import albumentations as A

IMG_SIZE = 512


class ResizeWithPad(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """

    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img):
        h_org, w_org, _ = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(
            1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org
        )
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh: resize_h + dh, dw: resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        return image


class DukeDataset(Dataset):
    
    def __init__(self, file_dir, transform=None):
        self.file_list = glob.glob(os.path.join(file_dir, "*", "*.png"))
        # print(len(self.file_list))
        self.transform = transform
        self.id_list = os.listdir(os.path.join(file_dir))

    def __getitem__(self, index):
        # img_path, label = self.file_list[index].split(',')
        # print(img_path)
        img_path = self.file_list[index]
        label = self.file_list[index].split('/')[-2]
        
        img = cv2.imread(img_path)
        idx_label = dict([(v, i) for i, v in enumerate(self.id_list)])
        label = idx_label[label.rstrip('\n')]
        label = np.array([label], dtype=np.int32)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img_tensor = ResizeWithPad((IMG_SIZE, IMG_SIZE))(img)
        img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).long()

        return img_tensor, label

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    pass
