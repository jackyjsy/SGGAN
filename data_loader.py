import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    # print(y)
    # print(y.size())
    y=np.asarray(y)
    # print(type(y))
    y=np.eye(num_classes, dtype='uint8')[y]
    return y

class CelebDataset(Dataset):
    def __init__(self, image_path, seg_path, metadata_path, transform, transform_seg1, transform_seg2, mode):
        self.image_path = image_path
        self.seg_path = seg_path
        self.transform = transform
        self.transform_seg1 = transform_seg1
        self.transform_seg2 = transform_seg2
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)

            if (i+1) < 2000:
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
            seg = Image.open(os.path.join(self.seg_path, self.train_filenames[index][:-3]+'png'))
            label = self.train_labels[index]
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
            seg = Image.open(os.path.join(self.seg_path, self.train_filenames[index][:-3]+'png'))
            label = self.test_labels[index]
        seg = self.transform_seg1(seg)
        
        num_s = 7
        seg_onehot = to_categorical(seg, num_s)
        seg=np.asarray(seg,dtype=np.long)
        return self.transform(image), torch.LongTensor(seg), self.transform_seg2(seg_onehot)*255.0, torch.FloatTensor(label)

    def __len__(self):
        return self.num_data


def get_loader(image_path, seg_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == 'CelebA':
        dataset = CelebDataset(image_path, seg_path, metadata_path, transform, transform_seg1, transform_seg2, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_path, transform)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader