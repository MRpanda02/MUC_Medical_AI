# -*- coding:utf-8 -*-

import os
import pickle
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer     # 原始文本转化为tf-idf的特征矩阵
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

seed = 222
random.seed(seed)
np.random.seed(seed)

def load_data(data_path):
    X, y = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            text, label = line.strip().split(',')
            # 每个字之间为' '
            text = ' '.join(list(text.lower()))
            X.append(text)
            y.append(label)

    index = np.arange(len(X))
    np.random.shuffle(index)
    X = [X[i] for i in index]
    y = [y[i] for i in index]
    return X, y

def run(data_path, model_save_path):
    X, y = load_data(data_path)
    # for i in range(len(X)):
    #     print(f'{X[i]}:{y[i]}')
    label_set = sorted(list(set(y)))
    # {'accept': 0, 'deny': 1, 'diagnosis': 2, 'goodbye': 3, 'greet': 4, 'isbot': 5}
    label2id = {label:idx for idx, label in enumerate(label_set)}
    # {0: 'accept', 1: 'deny', 2: 'diagnosis', 3: 'goodbye', 4: 'greet', 5: 'isbot'}
    id2label = {idx:label for label, idx in label2id.items()}

    # y是label集(label2id后)
    y = [label2id[i] for i in y]

    label_names = sorted(label2id.items(), key = lambda kv:kv[1], reverse = False)
    target_names = [i[0] for i in label_names]
    labels = [i[1] for i in label_names]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state=42)
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=0, max_df=0.9, analyzer='char', use_idf=1, smooth_idf=1,
                          sublinear_tf=1)
    train_X = vec.fit_transform(train_X)
    test_X = vec.transform(test_X)

    # ----------------GBDT-----------------
    gbdt = GradientBoostingClassifier(n_estimators=450, learning_rate=0.01, max_depth=8, random_state=24)
    gbdt.fit(train_X, train_y)
    train_new_feature_train = gbdt.apply(train_X)
    train_new_feature_train = train_new_feature_train.reshape(train_X.shape[0], -1)
    train_new_feature_test = gbdt.apply(test_X)
    train_new_feature_test = train_new_feature_test.reshape(test_X.shape[0], -1)
    print(train_new_feature_train.shape)
    print(train_new_feature_test.shape)
    enc = OneHotEncoder()
    enc.fit(train_new_feature_train)
    train_new_feature_train = np.array(enc.transform(train_new_feature_train).toarray())
    train_new_feature_test = np.array(enc.transform(train_new_feature_test).toarray())
    pred = gbdt.predict(test_X)
    print(classification_report(test_y, pred,target_names=target_names))
    print(confusion_matrix(test_y, pred,labels=labels))

    # -----------------LR------------------
    LR = LogisticRegression(C=8, dual=False, max_iter=400, n_jobs=2, multi_class='ovr', random_state=122)
    print(train_new_feature_train.shape)
    print(train_new_feature_test.shape)
    LR.fit(train_new_feature_train, train_y)
    pred = LR.predict(train_new_feature_test)
    print(classification_report(test_y, pred, target_names=target_names))
    print(confusion_matrix(test_y, pred,labels=labels))

    # # -------------融合--------------
    # pred_prob1 = LR.predict_proba(test_X)
    # pred_prob2 = gbdt.predict_proba(test_X)
    #
    # pred = np.argmax((pred_prob1 + pred_prob2)/2, axis=1)
    # print(classification_report(test_y, pred, target_names=target_names))
    # print(confusion_matrix(test_y, pred, labels=labels))

    pickle.dump(id2label, open(os.path.join(model_save_path, 'id2label.pkl'), 'wb'))
    pickle.dump(vec, open(os.path.join(model_save_path, 'vec.pkl'), 'wb'))
    pickle.dump(LR, open(os.path.join(model_save_path, 'LR.pkl'), 'wb'))
    pickle.dump(gbdt, open(os.path.join(model_save_path, 'gbdt.pkl'), 'wb'))

if __name__ == '__main__':
    run('intent_recog_data.txt','../model_file/')