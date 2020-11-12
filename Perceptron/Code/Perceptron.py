# encoding=utf-8
# @Author: Su
# @Date:   2020.11.12
# @Email:  tianyu.su@stu.pku.edu.cn

import pandas as pd
import numpy as np
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):
    def __init__(self):
        self.max_iteration = 500
        self.alpha = 0.00001

    def train(self, train_features, train_labels):
        self.w, self.b = [0.0 for i in range(len(train_features[0]))], 0

        seq_correct_time = 0
        while True:
            # 随机取一个点index
            index = random.randint(0, len(train_features) - 1)
            # 由于数据集中的类由0和1区分所以要将0和1映射到-1和1上
            x, y = list(train_features[index]), 2 * train_labels[index] - 1

            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            if (wx + self.b) * y > 0:
                # 对index点预测正确
                seq_correct_time += 1
                if seq_correct_time > self.max_iteration:
                    break
            else:
                # 对index点预测错误 更新w和b
                for i in range(len(self.w)):
                    self.w[i] += x[i] * y * self.alpha
                self.b += self.alpha * y
                seq_correct_time = 0


    def predict(self, test_features):
        predict_labels = []
        for feature in list(test_features):
            t = int((sum([self.w[j] * feature[j] for j in range(len(self.w))]) + self.b) > 0)
            predict_labels.append(t)
        return predict_labels


p = Perceptron()
print("start reading data...")
time1 = time.time()

# pixel0-pixel783是784个特征 label是分类结果1/0
raw_data = pd.read_csv("../Dataset/train_binary.csv", header=0)
data = raw_data.values
# 第一列是类别 后784列是特征
features, labels = data[0:, 1:], data[0:, 0]
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.33, random_state=23323)

time2 = time.time()
print("reading data uses", round(time2 - time1, 3), "second")

print("start training...")
time1 = time.time()
p.train(train_features, train_labels)
time2 = time.time()
print("training uses", round(time2 - time1, 3), "second")

print("start predicting...")
time1 = time.time()
test_predict = p.predict(test_features)
time2 = time.time()
print("predicting uses", round(time2 - time1, 3), "second")

score = accuracy_score(test_labels, test_predict)
print("The accuracy score is ", round(score, 3))
exit(0)
