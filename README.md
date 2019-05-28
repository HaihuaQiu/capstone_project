# 项目说明

## 本项目由于采用了深层网络模型，因此很难跑在本地电脑上，所有采用了Google免费提供的开源工具colab进行训练。

## 由于udacity发送文件大小的限制，以及作者的git lfs总存储量已经超过免费额度1G的缘故，故不提供数据集，数据集的下载网址为：
## https://github.com/udacity/cn-machine-learning/tree/master/mathematical_expression_recognition
## 数据集下载成功后解压缩到data文件夹中，这样data文件夹里具有一个有10万张图片training名称的文件夹和一个traning.csv标签文件。这样就可以运行代码里的make_tfrecord()函数制作训练集、验证集和测试集的tfrecord格式文件。

## 项目的组成文件：
### 1.crnn_model.py:是构建模型，并训练模型的文件
### 2.dataset.util.py:是把10万张图片和标签压缩成tfrecord并读取tfrerecord文件到模型中训练的文件。
### 3.util.py:是存储一些项目中用到的辅助函数的文件。
### 4.running_test.ipynb:是运行整个项目的文件。
### 5.data(缺）:是放置tfrecord数据集的文件夹。上传时，把10万张图片和标签删了，以提高上传效率。如果需要，可以自行下载training图片文件夹和traning.csv标签文件到data文件中。
### 6.saved_model:是放置训练好的模型的文件夹。
### 7.test_result:是放置训练过程中产生的数据的文件夹。

## 项目用到的包:
### 本地运行参考项目附件的 environment.yaml文件，云端运行直接使用colab。

## 机器硬件，机器操作系统
### 本地运行的硬件采用了GTX960M，系统采用win10,特别慢。 云端运行直接使用colab。

## 训练时间
### 使用colab运行了10个小时左右。

## 项目具体的操作：
### 1.把本项目上传到Google drive上。
### 2.选择项目中的runtest.ipynb以colaboratory的方式打开。
### 3.第一单元格是把环境变量变为Google drive上的路径，好运行项目环境和相关的包。如果在本地运行，则不需要此步骤。
### 4.接下来按顺序运行程序。
### 5.注意：
#### 5.1 make_tfrecord()必须在data文件夹中存在training图片文件夹和traning.csv标签文件才能运行，这是为了把10万张图片和标签压缩成tfrecord的代码。
#### 5.2 train函数有三种模式：
    5.2.1 index=0:表示从零开始训练模型。
    5.2.2 index=1:表示接着saved_model中的模型继续训练。
    5.2.3index=2:表示只评估saved_model中的模型性能而不做任何训练。
