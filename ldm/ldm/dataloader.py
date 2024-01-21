import random
import torchvision.transforms as T

from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
from pathlib import Path
import csv
import yaml

IMAGE_SIZE = 128

class ComposeState(T.Compose):
    def __init__(self, transforms):
        self.transforms = []
        self.mask_transforms = []

        for t in transforms:
            self.transforms.append(t)

        self.seed = None
        self.retain_state = False

    def __call__(self, x):
        # if self.seed is not None:   # retain previous state
        #     set_global_seed(self.seed)
        # if self.retain_state:    # save state for next call
        #     self.seed = self.seed or torch.seed()
        #     set_global_seed(self.seed)
        # else:
        #     self.seed = None    # reset / ignore state

        if isinstance(x, (list, tuple)):
            return self.apply_sequence(x)
        else:
            return self.apply_img(x)

    def apply_img(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def apply_sequence(self, seq):
        self.retain_state=True
        seq = list(map(self, seq))
        self.retain_state=False
        return seq

def cycle(dl):
    while True:
        for data in dl:
            yield data

class RandomRotate90():  # Note: not the same as T.RandomRotation(90)
    def __call__(self, x):
        x = x.rot90(random.randint(0, 3), dims=(-1, -2))
        return x

    def __repr__(self):
        return self.__class__.__name__
    
def dataset_to_list(data_path: str):
    # Get the list of class names present in the dataset
    files = [str(f).split("/")[-1].split(".")[0] for f in Path(data_path).iterdir() if f.name.endswith('jpg')]
    return files

def split_dataset(data_path: str, train_size: float = 0.9):
    """
    Splits a dataset into training and test sets, with each set containing data for each class.
    :param data_path: the path to the dataset
    :param train_size: the proportion of the data to use for training
    :return: two dictionaries, one containing the training data and the other containing the test data
    """

    list = dataset_to_list(data_path)
    random.shuffle(list)
    return list[:int(len(list) * train_size)], list[int(len(list) * train_size):]


def import_dataset(
        data_path: str = "",
        batch_size: int = 32,
        num_workers: int = 0,
        subclasses: list = [0, 1, 2, 3, 4, 5],
        cond_drop_prob: float = 0.5,
        threshold: float = 0.,
        force: bool = False,
        transform=None,
        **kwargs
):
    train_list, test_list = split_dataset(data_path, train_size=0.9)

    # Create the train and test datasets
    train_set = DatasetLung(data_path=data_path, file_list=train_list, 
                            transform=transform)
    test_set = DatasetLung(data_path=data_path, file_list=test_list, 
                           transform=transform)
    
    print(train_set)
    print(test_set)

    # Create the train and test data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


class DatasetLung(Dataset):
    def __init__(self,
            data_path: str,
            file_list: list[str],
            transform = None):

        # for extra in extra_unknown_data_path:
        #     data_dict = add_unconditional(data_path=extra, 
        #                                   data_dict=data_dict, no_check=True)
        self.file_list = file_list
        self.data_path = data_path
        self.transform = transform

    def __repr__(self):
        rep = f"{type(self).__name__}: ImageFolderDataset[{self.__len__()}]"
        return rep

    def __len__(self):
        return len(self.file_list)

    def multi_to_single_mask(self, mask):
        # Note to self - it seems we can return an empty mask (all zeros) for unlabelled data
        # mask=(mask*255).int()
        # if self.tmp_index==0:
        #     mask=torch.zeros_like(mask)
        # elif self.tmp_index==len(self.subclasses)+1:
        #     uniques=torch.unique(mask).int().tolist()
        #     uniques=[unique for unique in uniques if unique not in self.subclasses]
        #     if 0 in uniques:
        #         uniques.remove(0)
        #     for unique in uniques:
        #         mask=torch.where(mask==unique, -1, mask)
        #     mask=torch.where(mask!=-1, len(self.subclasses)+1, 0)
        # else:
        #     mask=torch.where(mask==self.subclasses[self.tmp_index-1], self.tmp_index, 0)
        return mask

    def unbalanced_data(self):
        core_path = str(random.choice(self.file_list))
        img_path = os.path.join(self.data_path, core_path+'.jpg')
        mask_path = os.path.join(self.data_path, core_path+'_mask.png')

        if not os.path.exists(img_path):
            for extra in self.extra:
                extra_path = os.path.join(extra, core_path+'.jpg')
                if os.path.exists(extra_path):
                    img_path = extra_path

        # load img and mask
        img = Image.open(img_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            h,w,c=np.array(img).shape
            mask=np.zeros((h,w,1))

        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE)).getchannel("R")

        return img,mask


    def __getitem__(self,idx):

        img, mask = self.unbalanced_data()

        if self.transform is not None:
            img,mask = self.transform((img,mask))

        # mask = self.multi_to_single_mask(mask)

        # print(img.shape)

        return img,mask
