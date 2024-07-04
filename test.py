import torch

# 检查GPU是否可用
if torch.cuda.is_available():
    print("PyTorch可以使用GPU")
    # 获取GPU数量
    print(f"GPU数量: {torch.cuda.device_count()}")
    # 获取第一个GPU的名字
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch无法使用GPU")
    

print(torch.cuda.is_available())
 
num_gpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())

print(torch.backends.cudnn.version())