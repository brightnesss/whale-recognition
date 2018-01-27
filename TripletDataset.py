import torch
import torch.utils.data
import pickle as pkl
import pandas as pd
import numpy as np
import os
import tqdm
from PIL import ImageEnhance
from PIL import Image

from utils import generate_triplets


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, n_triplets, transform=None, transform_parameter=np.array([1.0, 1.0, 1.0]), eval=False,
                 ratio=0.8):
        super(TripletDataset, self).__init__()
        self.dir = data_dir
        self.n_triplets = n_triplets
        self.transform = transform
        self.transform_parameter = transform_parameter
        self.eval = eval
        with open('LabelEncoder.pkl', 'rb') as f:
            self.le = pkl.load(f)
        self.train_csv = pd.read_csv('/data1/whale/train.csv').as_matrix()
        np.random.shuffle(self.train_csv)
        self.data = os.listdir(self.dir)

        print('Generating {} triplets'.format(self.n_triplets))
        self.triplets = generate_triplets(self.train_csv, self.le, self.n_triplets, len(self.le.classes_))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        anchor, positive, negative, label_positive, label_negative = self.triplets[item]

        anchor_dir = anchor.split('+')[0]
        anchor_dir = os.path.join(self.dir, anchor_dir)
        anchor_mode = anchor.split('+')[1]

        positive_dir = positive.split('+')[0]
        positive_dir = os.path.join(self.dir, positive_dir)
        positive_mode = positive.split('+')[1]

        negative_dir = positive.split('+')[0]
        negative_dir = os.path.join(self.dir, negative_dir)
        negative_mode = positive.split('+')[1]

        anchor_img = Image.open(anchor_dir).convert('RGB')
        if anchor_mode == 'F':
            anchor_img = anchor_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif anchor_mode == 'C':
            imgenhance_color = ImageEnhance.Color(anchor_img)
            anchor_img = imgenhance_color.enhance(np.random.random() * self.transform_parameter[0])
        elif anchor_mode == 'B':
            imgenhancer_bri = ImageEnhance.Brightness(anchor_img)
            anchor_img = imgenhancer_bri.enhance(np.random.random() * self.transform_parameter[1])
        elif anchor_mode == 'D':
            imgenhancer_contrast = ImageEnhance.Contrast(anchor_img)
            anchor_img = imgenhancer_contrast.enhance(np.random.random() * self.transform_parameter[2])

        positive_img = Image.open(positive_dir).convert('RGB')
        if positive_mode == 'F':
            positive_img = positive_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif positive_mode == 'C':
            imgenhance_color = ImageEnhance.Color(positive_img)
            positive_img = imgenhance_color.enhance(np.random.random() * self.transform_parameter[0])
        elif positive_mode == 'B':
            imgenhancer_bri = ImageEnhance.Brightness(positive_img)
            positive_img = imgenhancer_bri.enhance(np.random.random() * self.transform_parameter[1])
        elif positive_mode == 'D':
            imgenhancer_contrast = ImageEnhance.Contrast(positive_img)
            positive_img = imgenhancer_contrast.enhance(np.random.random() * self.transform_parameter[2])

        negative_img = Image.open(negative_dir).convert('RGB')
        if negative_mode == 'F':
            negative_img = negative_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif negative_mode == 'C':
            imgenhance_color = ImageEnhance.Color(negative_img)
            negative_img = imgenhance_color.enhance(np.random.random() * self.transform_parameter[0])
        elif negative_mode == 'B':
            imgenhancer_bri = ImageEnhance.Brightness(negative_img)
            negative_img = imgenhancer_bri.enhance(np.random.random() * self.transform_parameter[1])
        elif negative_mode == 'D':
            imgenhancer_contrast = ImageEnhance.Contrast(negative_img)
            negative_img = imgenhancer_contrast.enhance(np.random.random() * self.transform_parameter[2])

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, label_positive, label_negative
