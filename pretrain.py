# 基于ImageNet的预训练模型

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import time
import copy
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

data_dir = "/root/CV_2_lab1/lab2/processed_data"
save_path = "best_pr_model.pth"
log_dir = os.path.join("runs", "caltech101_classification")
os.makedirs(log_dir, exist_ok = True)
writer = SummaryWriter(log_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_dataset(pt_path):
    images, labels = torch.load(pt_path)
    return TensorDataset(images, labels)

train_dataset = load_dataset(os.path.join(data_dir, "train_data.pt"))
val_dataset = load_dataset(os.path.join(data_dir, "val_data.pt"))
test_dataset = load_dataset(os.path.join(data_dir, "test_data.pt"))

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 2)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False, num_workers = 2)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False, num_workers = 2)

dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
dataset_sizes = {x: len(dataloaders[x].dataset) for x in dataloaders}

_, train_labels = torch.load(os.path.join(data_dir, "train_data.pt"))
num_classes = len(set(train_labels.tolist()))
print(f"number of classes: {num_classes}")

model_ft= models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_ft.fc = nn.Linear(model_ft.fc.in_features,num_classes)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD([
	{"params": model_ft.fc.parameters(), "lr": 0.01},
	{"params": [p for n, p in model_ft.named_parameters() if "fc" not in n], "lr": 0.001}
], momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size = 6, gamma = 0.1)

#训练函数及曲线
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	since = time.time()

	train_loss_list = []
	val_loss_list = []
	train_acc_list = []
	val_acc_list = []

	for epoch in range(num_epochs):
		print(f"Epoch {epoch+1}/{num_epochs}")
		print("-" * 20)

		for phase in ["train", "val"]:
			model.train() if phase =="train" else model.eval()

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloaders[phase]:
				inputs, labels = inputs.to(device), labels.to(device)

				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == "train"):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)
					if phase == "train":
						loss.backward()
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			if phase == "train":
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
			writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)

			if phase == "train":
				train_loss_list.append(epoch_loss)
				train_acc_list.append(epoch_acc.item())
			else:
				val_loss_list.append(epoch_loss)
				val_acc_list.append(epoch_acc.item())

			print(f"{phase.capitalize()} Loss:{epoch_loss:.4f} Acc: {epoch_acc:.4f}")

			if phase == "val" and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
				torch.save(best_model_wts, save_path)
				print("Model improved and saved!")

	elapsed = time.time() - since
	print(f"Training completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
	print(f"Best validation accuracy: {best_acc:.4f}")

	model.load_state_dict(best_model_wts)

	plot_curves(train_loss_list, val_loss_list, train_acc_list, val_acc_list)

	return model

def plot_curves(train_loss, val_loss, train_acc, val_acc):
	epochs = range(1, len(train_loss) +1)

	plt.figure(figsize=(12, 5))
	plt.subplot(1,2,1)
	plt.plot(epochs, train_loss, label='Train Loss')
	plt.plot(epochs, val_loss, label='Val Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title("Loss Curve")
	plt.legend()

	plt.subplot(1,2,2)
	plt.plot(epochs, train_acc, label='Train Accuracy')
	plt.plot(epochs, val_acc, label="Val Accuracy")
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Accuracy Curve')
	plt.legend()

	plt.tight_layout()
	plt.savefig("training_curves.png")
	plt.show()
	print("Training curves saved as 'training_curves.png'.")

def evaluate_model(model, dataloader):
	model.eval()
	corrects = 0
	total = 0

	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			corrects += torch.sum(preds == labels)
			total += labels.size(0)

	return (corrects.double() / total).item()

if __name__ == "__main__":
	print("Start training...")
	model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs = 20)

	print("Evaluating best model on test set...")
	test_acc = evaluate_model(model_ft, test_loader)
	print(f"Test Accuracy: {test_acc:.4f}")
