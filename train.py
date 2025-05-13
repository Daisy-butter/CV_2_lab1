import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 数据路径
data_dir = "/root/CV_2_lab1/lab2/processed_data"
save_path = "best_caltech_model.pth"

# 加载数据函数
def load_dataset(pt_path):
    images, labels = torch.load(pt_path)
    return TensorDataset(images, labels)

# 加载数据集
train_dataset = load_dataset(os.path.join(data_dir, "train_data.pt"))
val_dataset   = load_dataset(os.path.join(data_dir, "val_data.pt"))
test_dataset  = load_dataset(os.path.join(data_dir, "test_data.pt"))

# 获取类别数
_, train_labels = torch.load(os.path.join(data_dir, "train_data.pt"))
num_classes = len(set(train_labels.tolist()))
print(f"Number of classes: {num_classes}")

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# 定义模型（从零开始训练）
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            total += labels.size(0)
    return (running_corrects.double() / total).item()

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=10):
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        val_acc = evaluate_model(model, val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
              f"Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc.item())
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("Model saved.")

    return history

# 可视化函数
def plot_training_curves(history, filename="training_curve_train.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker='o', label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker='o', label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], marker='s', label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Training curves saved to {filename}")
    plt.close()

# 主函数
if __name__ == "__main__":
    print("Start training...")
    history = train_model(model, train_loader, val_loader, num_epochs=30)

    # 可视化
    plot_training_curves(history)

    # 测试模型
    model.load_state_dict(torch.load(save_path))
    test_acc = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")
