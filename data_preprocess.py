# 预处理目标数据集

import os
from PIL import Image
import random
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
import torch

# 定义数据集路径
DATASET_PATH = "/root/CV_2_lab1/lab2/caltech-101/101_ObjectCategories/101_ObjectCategories"

# 设置随机种子以确保可复现
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# 图像数据集类定义
class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 读取目录并构造路径与类别标签
        for idx, category in enumerate(os.listdir(root_dir)):  # 类别文件夹
            category_dir = os.path.join(root_dir, category)
            if os.path.isdir(category_dir):  # 确保是文件夹
                for filename in os.listdir(category_dir):
                    file_path = os.path.join(category_dir, filename)
                    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 仅加载图片
                        self.image_paths.append(file_path)
                        self.labels.append(idx)  # 使用类别索引作为标签
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # 确保图像是 RGB 模式
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据增强 & 标准化转换（符合 ImageNet 标准）
def get_transforms(dataset_type='train', size=224):
    if dataset_type == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(size),                # 随机裁剪并调整大小
            transforms.RandomHorizontalFlip(p=0.5),           # 随机水平翻转
            transforms.RandomRotation(15),                   # 随机旋转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准化
                                 std=[0.229, 0.224, 0.225])
        ])
    else:  # 'val' or 'test'
        return transforms.Compose([
            transforms.Resize((size, size)),                  # 调整到固定大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准化
                                 std=[0.229, 0.224, 0.225])
        ])

# 划分数据集
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])

# 保存处理后的数据（可选，保存为 .pt 文件）
def save_partitioned_data(dataset, transform, save_dir, partition_name, batch_size=32):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{partition_name}_data.pt')
    
    # 应用转换
    dataset.dataset.transform = transform
    
    images = []
    labels = []
    for img, label in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images.append(img)
        labels.append(label)

    # 合并张量
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # 保存为张量
    torch.save((images, labels), save_path)
    print(f"{partition_name} data saved to {save_path}")

if __name__ == "__main__":
    # 定义保存路径
    output_dir = os.path.join("/root/CV_2_lab1/lab2", 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建数据集
    full_dataset = Caltech101Dataset(root_dir=DATASET_PATH, transform=None)

    # 划分训练集、验证集和测试集
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    
    # 保存数据
    save_partitioned_data(train_dataset, get_transforms('train'), output_dir, 'train')
    save_partitioned_data(val_dataset, get_transforms('val'), output_dir, 'val')
    save_partitioned_data(test_dataset, get_transforms('test'), output_dir, 'test')

    print("Data preprocessing completed!")