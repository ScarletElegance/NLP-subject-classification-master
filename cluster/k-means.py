import random
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import multiprocessing
from collections import Counter

TaggededDocument = gensim.models.doc2vec.TaggedDocument
#加载停用词
# stopwords=pd.read_csv('D://input_py//day07//stopwords.txt',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
# stopwords=stopwords['stopword'].values
#加载语料
# laogong_df = pd.read_csv('D:/NLP-master/fenji/true_data1261.txt', encoding='utf-8', sep=',')
result_text_path = 'D:/NLP-master/fenji/result.txt'
with open('D:/NLP-master/fenji/true_data1261.txt', 'r', encoding='utf-8') as cf:
    sentences = cf.readlines()
# print(laogong_df)
# laopo_df = pd.read_csv('D://input_py//day07//beilaogongda.csv', encoding='utf-8', sep=',')
# erzi_df = pd.read_csv('D://input_py//day07//beierzida.csv', encoding='utf-8', sep=',')
# nver_df = pd.read_csv('D://input_py//day07//beinverda.csv', encoding='utf-8', sep=',')
#删除语料的nan行
# laogong_df.dropna(inplace=True)
# laopo_df.dropna(inplace=True)
# erzi_df.dropna(inplace=True)
# nver_df.dropna(inplace=True)
#转换
# sentences = laogong_df.tolist()
# laopo = laopo_df.segment.values.tolist()
# erzi = erzi_df.segment.values.tolist()
# nver = nver_df.segment.values.tolist()

# 定义分词函数preprocess_text
def preprocess_text(content_lines, sentences):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]#去数字
            segs = list(filter(lambda x:x.strip(), segs))   #去左右空格
            segs = list(filter(lambda x:len(x)>1, segs)) #长度为1的字符
            segs = list(filter(lambda x:x not in stopwords, segs)) #去掉停用词
            sentences.append(" ".join(segs))
        except Exception:
            print(line)
            continue

# sentences = []
# preprocess_text(laogong, sentences)
# preprocess_text(laopo, sentences)
# preprocess_text(erzi, sentences)
# preprocess_text(nver, sentences)

random.shuffle(sentences)
# 控制台输出前10条数据
for sentence in sentences[:10]:
    print(sentence)

# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences))
# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
# 查看特征大小
print ('Features length: ' + str(len(word)))

# TF-IDF 的中文文本 K-means 聚类
numClass=3  # 聚类分几簇
clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  #这里也可以选择随机初始化init="random"
pca = PCA(n_components=10)  # 降维
TnewData = pca.fit_transform(weight)  # 载入N维
s = clf.fit(TnewData)

# 定义聚类结果可视化函数
def plot_cluster(result,newData,numClass):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
             'g^'] * 3
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            # print ind1
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, color[i])

    # 绘制初始中心点
    x1 = []
    y1 = []
    for ind1 in clf.cluster_centers_:
        try:
            y1.append(ind1[1])
            x1.append(ind1[0])
        except:
            pass
    plt.plot(x1, y1, "rv") #绘制中心
    plt.show()

# 对数据降维到2维，绘制聚类结果图
# pca = PCA(n_components=2)  # 输出2维
# newData = pca.fit_transform(weight)  # 载入N维
# result = list(clf.predict(TnewData))
# plot_cluster(result,newData,numClass)

# 先用 PCA 进行降维，再使用 TSNE
from sklearn.manifold import TSNE
newData = PCA(n_components=4).fit_transform(weight)  # 载入N维
newData =TSNE(2).fit_transform(newData)
result = list(clf.predict(TnewData))
# num_Count=Counter(result)
# print(num_count)
num_count={}
for i in result:
    if i not in num_count:
        num_count[i]=1
    else:
        num_count[i]+=1
print(num_count)
plot_cluster(result,newData,numClass)
def get_datasest():
    with open('D:/NLP-master/fenji/true_data1261.txt', 'r', encoding = 'utf-8') as cf:
        docs = cf.readlines()
        print
        len(docs)

    x_train = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        # 训练模型前，先将语料整理成规定的形式，这里用到TaggedDocument模型
        # 输入输出内容都为 词袋 + tag列表
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)

    return x_train

x_train = get_datasest()
with open(result_text_path, 'w') as wf:
    i = 0
    while i < len(x_train):
        string = ""
        text = x_train[i][0]
        for word in text:
            string = string + word
        string = string + '\t'
        string = string + str(result[i])
        string = string + '\n'
        wf.write(string)
        i = i + 1