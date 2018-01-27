import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable, Function
import pickle as pkl
from tqdm import tqdm
import os
import numpy as np
import pandas as pd


class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)


def generate_triplets(train_csv, le, num_triplets, n_classes):
    # generate triplets set
    def create_indices(_train_csv, _le):
        inds = dict()
        for i in range(len(_train_csv)):
            img = _train_csv[i][0]
            label = _le.transform([_train_csv[i][1]])[0]
            if label not in inds:
                inds[label] = []
            # original image
            inds[label].append(img + '+O')
            # flip left and right using Image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            inds[label].append(img + '+F')
            # enhance color using ImageEnhance.Color
            # imgenhance_color = ImageEnhance.Color(img)
            # img_color = imgenhance_color.enhance(np.random.random()*max_value)
            inds[label].append(img + '+C')
            # enhance brightness
            # imgenhancer_bri = ImageEnhance.Brightness(img)
            inds[label].append(img + '+B')
            # enhance contrast
            # imgenhancer_contrast=ImageEnhance.Contrast(img)
            inds[label].append(img + '+D')
        return inds

    triplets = []
    # Indices = array of labels and each label is an array of indices
    indices = create_indices(train_csv, le)

    for x in tqdm(range(num_triplets)):
        label_positive = np.random.randint(0, n_classes)
        label_negative = np.random.randint(0, n_classes)

        while label_positive == label_negative:
            label_negative = np.random.randint(0, n_classes)

        n1 = np.random.randint(0, len(indices[label_positive]))
        n2 = np.random.randint(0, len(indices[label_positive]))
        while n1 == n2:
            n2 = np.random.randint(0, len(indices[label_positive]))
        if len(indices[label_negative]) == 1:
            n3 = 0
        else:
            n3 = np.random.randint(0, len(indices[label_negative]))

        triplets.append(
            [indices[label_positive][n1], indices[label_positive][n2], indices[label_negative][n3], label_positive,
             label_negative])
    return triplets


if __name__ == '__main__':
    # generate triplets
    train_csv = pd.read_csv('/data1/whale/train.csv').as_matrix()
    with open('LabelEncoder.pkl', 'rb') as f:
        le = pkl.load(f)
    num_triplets = 1000000
    n_classes = 4251
    generate_triplets(train_csv, le, num_triplets, n_classes)
