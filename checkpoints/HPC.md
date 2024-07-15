用来保存checkpoints的目录，内用显卡和轮次命名

**本文件用于记录每次训练的关键更改**

# A800

## 1
开启了数据增强
大约在60轮趋于稳定

## 3
为mask开启了形态变换的数据增强，和images同步 
(其实失败了 没同步)

epochs = 150
LR=0.1
Batch_Size=80
Num_workers=8

## 4 （取消
为mask开启了形态变换的数据增强，和images同步
使用大哥的高质量数据集

## 6
12边界分割
使用b2模型

## 7

12边界 使用b0模型
IS_IOU_ACC=False