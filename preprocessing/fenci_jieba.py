# coding:utf-8
import jieba
import sys
import time

sys.path.append("../../")
import codecs
import os
import re


# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

def LoadStopWordList(filepath):
    """
    创建停用词list
    """
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

#定义分词函数
def FenCi(readfile, outfile, stopwords):
    # r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+0123456789'
    for line in readfile.readlines():
        # 更高效的字符串替换
        # line = re.sub(r, ' ', line)
        newline = jieba.cut(line, cut_all=False)
        outstr_list = list()
        for word in newline:
            if word not in stopwords:#同时去除停用词
                outstr_list.append(word)
        str_out = ' '.join(outstr_list)#分词用空格隔开
        # str_out.encode('utf-8')\
        print(str_out)
        print(str_out, file=outfile, end=' ')


if __name__ == '__main__':
    fromdir = "./data/"
    todir = "./fenci/"
    stopWordFile = "stop_words.txt"
    # 一次只能对一个文档进行分词
    # file = "T_train2949.txt"
    # file = "F_train2974.txt"
    file = "T_test1263.txt"
    # file = "F_test1274.txt"
    ofile = "jieba.txt"
    infile = open(os.path.join(fromdir, file), 'r', encoding='UTF-8')
    outfile = open(os.path.join(todir, ofile), 'w+', encoding='UTF-8')
    # 这里加载停用词
    stopwords = [line.strip() for line in open(os.path.join(todir, stopWordFile), 'r', encoding='UTF-8').readlines()]
    FenCi(infile, outfile, stopwords)
    infile.close()
    outfile.close()
