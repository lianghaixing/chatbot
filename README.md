# chatbot
基于深度学习的聊天机器人，使用tensorflow + RNN实现

###训练语料
网上公开的语料训练库--小黄鸡语料库，戳我(https://github.com/lianghaixing/Dialog_Corpus)

###文件夹介绍
* data--原始数据以及数据清洗后的字典表，转化为相应的向量输入，enc为encoder的输入，dec为decoder的输入
* result--测试集的问答效果，为对话形式

###代码介绍

* data_utils.py--数据清洗，将数据分为问句和答句
* prepareData.py--生成字典表，转化为数字向量，将中文输入转化为向量输入
* seq2seq_model.py--定义seq2seq模型，初始化函数，定义类
* execute.py--执行函数，定义训练部分，测试部分
* train.py--训练函数
* predict.py--测试函数，定义测试集的测试
* seq2seq.ini--配置函数，定义一些训练的参数
* getConfig.py--获取配置函数的方法

###训练环境

windows 10
python 3.6.1
tensorflow==1.2.1

###聊天效果图
![Image text](https://github.com/lianghaixing/chatbot/tree/master/picture/pic1.png)
![Image text](https://github.com/lianghaixing/chatbot/tree/master/picture/pic2.png)
###result测试内容

```
ask: 我还喜欢她,怎么办
answer: 我要努力了，我也知道

ask: 什么意思
answer: 不想哄哄的，主人有什么我很喜欢。。。。

ask: 你相公是谁
answer: 当然是美丽的主人咯

ask: 有
answer: 啊…我只有你……

ask: 看来,你是自动回复的
answer: 嗯
```