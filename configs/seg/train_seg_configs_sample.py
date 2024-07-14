#用于Git同步的配置文件。实际配置在train_seg_configs.py中
#注！Linux的路径是/ win的路径是\

import os.path as path

#任务代号
taskID=r'A800-3'
# 设置保存目录
out_weights_path = path.join('checkpoints',taskID)
#TensorBoard日志目录
TensorBoard_dir=path.join('tensorboard-log',taskID)

#数据集目录
root_dir = r'E:\项目数据\轮胎\分割数据集\总数据集\分割格式'

# 设置保存模型数量
max_to_save = 200  # 只保存最后n轮训练的模型
# 设置训练轮数
epochs = 130
# 学习率
LR=0.06
Batch_Size=8
Num_workers=6
#K折交叉验证
N_splits=5
#模型的大小
Model_size='b0'