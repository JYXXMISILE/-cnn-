import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义归一化变换
transform = transforms.Compose([
    transforms.ToTensor(),                # 转换为 [0, 1] 区间的张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

# 加载数据集并应用变换
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# 检查数据集的归一化
print(f"First train image tensor min: {train_data[0][0].min()}, max: {train_data[0][0].max()}")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


