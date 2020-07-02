# NLP-subject-classification-master
基于Text-CNN算法判断文本是否具有指定主题倾向的二分类器


------


## 项目概述

本项目使用Text-CNN算法实现一个判断文本是否具有指定主题倾向的二分类器。



## 目录结构

**【Weibo_Spider】**

* search_spider

  * hour_fenge.py
  * search_start.py
    

* tools

  * Cookie_Process.py
  * Date_Process.py
  * Emoji_Process.py
  * Number_Process.py
  * Weibo_Driver.py
  

**【classifier】**

* data
  
* runs
  
* data_helpers.py
* eval.py
* text_cnn.py
* train.py


**【cluster】**


* k-means.py


**【custom classifier】**

* predict_true.ipynb


**【preprocessing】**：

* fenci
* fenci_jieba.py
* pre_data.ipynb



## 版本管理

v 1.0.0




## 依赖配置

#### 1. Python环境

- Python 3.6及以上版本


#### 2. python 开发工具

* PyCharm


#### 3. 需要配置的python依赖包

- pandas-0.23.2-cp37-cp37m-win_amd64.whl
- numpy-1.16.6-cp37-cp37m-win_amd64.whl
- scipy-1.4.1-cp37-cp37m-win_amd64.whl
- scikit_learn-0.19.2-cp37-cp37m-win_amd64.whl
- gensim-3.8.3-cp36-cp36m-win_amd64.whl
- json5-0.9.3-py3-none-any.whl
- joblib-0.14.1-py2.py3-none-any.whl
  


## 部署说明

#### 1. Python环境

在Windows环境下推荐直接下载Anaconda完成Python所需环境的配置。

> 下载地址为：https://www.anaconda.com/。

#### 2. PyCharm安装及配置

详细过程可参考教程：https://blog.csdn.net/yang520java/article/details/80255659。

> 下载地址为：http://www.jetbrains.com/pycharm/download/#section=windows。

#### 3.本项目依赖的Python库安装说明

1）更新pip

```
python -m pip install --upgrade pip
```

2）下载库文件（以pandas为例）

在网址https://www.lfd.uci.edu/~gohlke/pythonlibs/ 中找到你需要的库文件版本。

例如windows 64 位 Python3.7 对应下载:pandas-1.0.3-cp37-cp37m-win_amd64.whl。下载后放置到Python的安装目录。

3）安装库文件（以pandas为例）

cmd进入终端，cd到Python的安装目录，即下载文件放置的目录，在终端输入如下命令：

```
pip install pandas-1.0.3-cp37-cp37m-win_amd64.whl
```



## 运行说明
### 一.  爬取数据

运行Weibo_Spider/search_spider/search_start.py


### 二.  文本预处理

先用Jupyter Notebook运行preprocessing/pre_data.ipynb
再运行preprocessing/fenci_jieba.py


### 三.  分类模型

#### 1. 训练模型

运行classifier/train.py

#### 2. 调用模型进行预测

运行classifier/eval.py


### 四.  拓展研究

先运行cluster/k-means.py
再用Jupyter Notebook运行custom classifier/predict_true.ipynb

## 注意事项

1. 运行代码时，将原路径更换为自己的路径，且路径名最好不含中文
2. 模型的特征维数和测试数据处理后的特征维数必须相等



