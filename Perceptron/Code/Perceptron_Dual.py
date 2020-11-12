# encoding=utf-8
# @Author: Su
# @Date:   2020.11.12
# @Email:  tianyu.su@stu.pku.edu.cn

import pandas as pd
import numpy as np
import time
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron_Dual:
    def __init__(self):
        self.max_iteration = 5
        self.alpha = 0.00001

    def compute_Gram(self, train_features, N):
        gram = [[-1 for j in range(N)] for i in range(N)]
        for i in range(N):
            for j in range(N):
                if gram[j][i] != -1:
                    gram[i][j] = gram[j][i]
                else:
                    gram[i][j] = sum(
                        [train_features[i][k] * train_features[j][k] for k in range(len(train_features[0]))])
        return gram

    def train(self, g, train_features, train_labels):
        self.n, self.b = [0 for i in range(len(train_features))], 0
        seq_correct_count = 0
        while True:
            index = random.randint(0, len(train_features) - 1)
            y = 2 * train_labels[index] - 1
            temp = 0
            for i in range(len(train_features)):
                temp += self.n[i] * g[i][index]
            temp = y * (temp * self.alpha + self.b)
            if temp > 0:
                seq_correct_count += 1
                if seq_correct_count > self.max_iteration:
                    break
                else:
                    self.n[index] += 1
                    self.b += self.alpha * y
                    seq_correct_count = 0

    def predict(self, test_features, train_features, train_labels, N):
        ans, w = [], [0 for i in range(len(test_features[0]))]
        for i in range(N):
            for j in range(len(w)):
                w[j] += self.n[i] * train_features[i][j] * self.alpha * train_labels[i]

        for feature in list(test_features):
            t = int((sum([w[j] * feature[j] for j in range(len(w))]) + self.b) > 0)
            ans.append(t)
        return ans


p_d = Perceptron_Dual()
print("start reading data...")
time1 = time.time()

# pixel0-pixel783是784个特征 label是分类结果1/0
raw_data = pd.read_csv("../Dataset/train_binary.csv", header=0)
data = raw_data.values
# 第一列是类别 后784列是特征
features, labels = data[41300:, 1:], data[41300:, 0]
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.33, random_state=23323)

time2 = time.time()
print("reading data uses", round(time2 - time1, 3), "second")

print("start computing Gram...")
g = p_d.compute_Gram(train_features, len(train_features))
print("computing Gram uses", round(time2 - time1, 3), "second")

print("start training...")
time1 = time.time()
p_d.train(g, train_features, train_labels)
time2 = time.time()
print("training uses", round(time2 - time1, 3), "second")

print("start predicting...")
time1 = time.time()
test_predict_label = p_d.predict(test_features, train_features, train_labels, len(train_features))
time2 = time.time()
print("predicting uses", round(time2 - time1, 3), "second")

score = accuracy_score(test_labels, test_predict_label)
print("The accuracy score is ", round(score, 3))
exit(0)
