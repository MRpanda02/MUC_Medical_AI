# -*- coding:utf-8 -*-

import pickle
import os

f = open(os.path.join('./model_file/', 'LR.pkl'), 'rb')
data = pickle.load(f)
print(data)

