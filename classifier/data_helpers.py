# coding:utf-8
import numpy as np
import re
import itertools
from collections import Counter
import sys
# import importlib
# importlib.reload(sys) # reload(sys)
# sys.setdefaultencoding("utf-8")


# 剔除英文的符号
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(T_data_file, F_data_file):
    """
    加载二分类训练数据，为数据打上标签
    (X,[0,0])
    X =  东方 财富 美国 中国 核电 公司 列入 实体 清单 中广核 回应 影响 可控 网页 链接 东方 财富
    Y = 美国实体清单
    0:其他---> [1,0]
    1:美国实体清单--->[0,1]
    (X,Y)
    """
    T_examples = list(open(T_data_file, "r", encoding="utf-8").readlines())
    T_examples = [s.strip() for s in T_examples]#删除空白符
    F_exampless = list(open(F_data_file, "r", encoding="utf-8").readlines())
    F_exampless = [s.strip() for s in F_exampless]
    x_text = T_examples + F_exampless
    
    # 适用于英文
    # x_text = [clean_str(sent) for sent in x_text]

    x_text = [sent for sent in x_text]
    # 定义类别标签 ，格式为one-hot的形式: y=1--->[0,1]
    positive_labels = [[0, 1] for _ in T_examples]
    # print positive_labels[1:3]
    negative_labels = [[1, 0] for _ in F_exampless]
    y = np.concatenate([positive_labels, negative_labels], 0)
    """
    print y
    [[0 1]
     [0 1]
     [0 1]
     ..., 
     [1 0]
     [1 0]
     [1 0]]
    print y.shape
    (10662, 2)
    """
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    利用迭代器从训练数据会取某一个batch的数据
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # 每回合打乱顺序
        if shuffle:
            # 随机产生以一个乱序数组，作为数据集数组的下标
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 划分批次
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# 测试代码用的
if __name__ == '__main__':
    jisuanji_data_file = './fenci/T_train2949.txt'
    jiaotong_data_file = './fenci/F_train2974.txt'
    load_data_and_labels(T_data_file, F_data_file)








