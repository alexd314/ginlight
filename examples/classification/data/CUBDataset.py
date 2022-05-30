import typing as tp
from torch.utils.data import Dataset, Subset
import torch
import torch.nn as nn
import PIL.Image as Image
import os

class CUBDataset(Dataset):

    def __init__(self, path_to_dataset : str, train_transform : tp.Union[None, nn.Module] = None, test_transform : tp.Union[None,nn.Module]= None):
        self._path_to_dataset = path_to_dataset
        self._train_transform = train_transform
        self._test_transform = test_transform
        self._setup = False

    def _read_images_index(self):
        images_txt_file_path = os.path.join(self._path_to_dataset,"images.txt")
        self._image_index = { }
        with open(images_txt_file_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                img_id, img_path = line.split(" ",1)
                img_id = int(img_id)-1                                                          # zero-based indexing of image ids
                self._image_index[img_id] = os.path.join(self._path_to_dataset, "images", img_path)

    def _read_image_labels(self):
        image_class_labels_txt_path = os.path.join(self._path_to_dataset,"image_class_labels.txt")
        self._image_class_labels = { }
        with open(image_class_labels_txt_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                img_id, img_label = line.split(" ",1)
                img_id = int(img_id)-1
                img_label = int(img_label)-1                         # zero based indexing of images and classes
                self._image_class_labels[img_id] = img_label

    def _read_train_test_splits(self):
        train_test_split_txt_path = os.path.join(self._path_to_dataset,"train_test_split.txt")
        self._train_images = set()
        self._test_images = set()
        with open(train_test_split_txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                imid, train_split = tuple(map(lambda x: int(x.rstrip()), line.split(" ")))
                if train_split == 1:
                    self._train_images.add(imid-1)           # zero based indexing of images
                else:
                    self._test_images.add(imid-1)

    def setup(self, stage : str):
        if not self._setup:
            self._read_images_index()
            self._read_image_labels()
            self._read_train_test_splits()
            self._setup = True
    def __len__(self):
        return len(self._image_index)

    def __getitem__(self, index : int) -> tp.Tuple[torch.Tensor, int]:
        class_id = self._image_class_labels[index]
        image_path = self._image_index[index]
        image = Image.open(image_path).convert('RGB')

        if index in self._train_images and self._train_transform is not None:
            image = self._train_transform(image)
        elif index in self._test_images and self._test_transform is not None:
            image = self._test_transform(image)

        return image, class_id

    def train_dataset(self):
        return Subset(self, list(sorted(self._train_images)))

    def test_dataset(self):
        return Subset(self, list(sorted(self._test_images)))

    def val_dataset(self):
        return self.test_dataset()