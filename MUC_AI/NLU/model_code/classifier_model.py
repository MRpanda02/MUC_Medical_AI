# -*- coding:utf-8 -*-

import os
import pickle
import numpy as np
from sklearn import svm

class Classi_Model(object):

    def __init__(self, model_save_path):
        super(Classi_Model, self).__init__()
        self.model_save_path = model_save_path
        self.id2label = pickle.load(open(os.path.join(self.model_save_path, 'id2label.pkl'), 'rb'))
        self.vec = pickle.load(open(os.path.join(self.model_save_path, 'vec.pkl'), 'rb'))
        self.LR = pickle.load(open(os.path.join(self.model_save_path, 'LR.pkl'), 'rb'))
        self.GBDT = pickle.load(open(os.path.join(self.model_save_path, 'gbdt.pkl'), 'rb'))

    def predict(self, text):
        text = ' '.join(list(text.lower()))
        text = self.vec.transform([text])
        proba1 = self.LR.predict_proba(text)
        proba2 = self.GBDT.predict_proba(text)
        label = np.argmax((proba1+proba2)/2, axis = 1)
        return self.id2label.get(label[0])

if __name__ == '__main__':
    model = Classi_Model('../model_file/')

    text = '请问心脏病怎么治'
    label = model.predict(text)
    print(label)


