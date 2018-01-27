import pandas as pd
import pickle as  pkl
from sklearn import preprocessing
import os

data_dir = '/data1/whale/train.csv'

data = pd.read_csv(data_dir)

le = preprocessing.LabelEncoder()
le.fit(data.Id)

le_dir = '~/whale-recognition/LabelEncoder.pkl'
le_dir = os.path.expanduser(le_dir)

with open(le_dir, 'wb') as f:
    pkl.dump(le, f)