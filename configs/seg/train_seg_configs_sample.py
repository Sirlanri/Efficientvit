#用于Git同步的配置文件。实际配置在train_seg_configs.py中
#注！Linux的路径是/ win的路径是\

#数据集目录
root_dir = r'E:\项目数据\轮胎\分割数据集\总数据集\分割格式'
# 设置保存目录
out_weights_path = r'checkpoints\0'
# 设置保存模型数量
max_to_save = 200  # 只保存最后n轮训练的模型
# 设置训练轮数
epochs = 130
# 学习率
LR=0.06
#Batch_Size
Batch_Size=8
#TensorBoard日志目录
TensorBoard_dir=r'tensorboard-log\5'

Num_workers=6