import torch
from torchvision import datasets, transforms

# 定义数据变换
transform = transforms.Compose([transforms.ToTensor()])

# 下载和加载训练数据集
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# 检查数据集大小
print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")