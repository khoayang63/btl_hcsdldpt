"""
train_leaf_model.py
Huấn luyện (Fine-tuning) mạng ResNet-18 trên tập dữ liệu 8 loài lá cây (1200 ảnh).
Lưu model checkpoint vào db/finetuned_resnet18_leaf.pth
Và xuất biểu đồ đường cong huấn luyện Loss/Accuracy vào visualization/07_training_curves.png
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_DIR = os.path.join(PROJECT_ROOT, "db")
VIZ_DIR = os.path.join(PROJECT_ROOT, "visualization")
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

class LeafDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def main():
    print("=" * 70)
    print("  HUẤN LUYỆN (FINE-TUNING) RESNET-18 TRÊN TẬP DỮ LIỆU LÁ CÂY")
    print("=" * 70)

    # 1. Tải metadata dataset
    meta_path = os.path.join(DATA_DIR, "dataset_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)

    filepaths = [item["filepath"] for item in dataset_meta]
    categories = [item["category"] for item in dataset_meta]
    unique_cats = sorted(list(set(categories)))
    cat_to_id = {c: i for i, c in enumerate(unique_cats)}
    id_to_cat = {i: c for c, i in cat_to_id.items()}
    labels = [cat_to_id[c] for c in categories]

    print(f"Tổng số ảnh: {len(filepaths)} ảnh | 8 loài: {', '.join(unique_cats)}")

    # 2. Phân chia tập Train (80% = 960 ảnh) và Validation (20% = 240 ảnh) có bảo toàn tỉ lệ loài (Stratified)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filepaths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Tập Huấn luyện (Train): {len(train_paths)} ảnh ({len(train_paths)//len(unique_cats)} ảnh/loài)")
    print(f"Tập Kiểm định (Val):     {len(val_paths)} ảnh ({len(val_paths)//len(unique_cats)} ảnh/loài)")

    train_tf, val_tf = get_transforms()
    train_dataset = LeafDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = LeafDataset(val_paths, val_labels, transform=val_tf)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Khởi tạo mô hình ResNet-18 Pretrained ImageNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nThiết bị huấn luyện: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Thay thế lớp fc cuối cùng cho 8 loài lá cây kèm Dropout phòng ngừa Overfitting
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(unique_cats))
    )
    model.to(device)

    # Thiết lập hàm mất mát và tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    # Huấn luyện end-to-end với tốc độ học vừa phải
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-2)
    epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Vòng lặp huấn luyện (Training Loop)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(DB_DIR, "finetuned_resnet18_leaf.pth")

    print("\n" + "-" * 70)
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^12} | {'Val Loss':^12} | {'Val Acc':^12} | {'Thời gian':^10}")
    print("-" * 70)

    start_total = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()

        # Phase 1: Train
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == targets.data).item()
            total_train += targets.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Phase 2: Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == targets.data).item()
                total_val += targets.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0
        scheduler.step()

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        epoch_sec = time.time() - t_epoch_start
        print(f"{epoch:^8d} | {epoch_train_loss:^12.4f} | {epoch_train_acc:^11.2f}% | {epoch_val_loss:^12.4f} | {epoch_val_acc:^11.2f}% | {epoch_sec:^9.1f}s")

        # Lưu model tốt nhất
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'categories': unique_cats,
                'cat_to_id': cat_to_id
            }, best_model_path)

    total_time = time.time() - start_total
    print("-" * 70)
    print(f"Hoàn thành huấn luyện trong {total_time:.1f}s! Độ chính xác cao nhất trên tập Val: {best_val_acc:.2f}%")
    print(f"Đã lưu checkpoint tốt nhất tại: {best_model_path}")

    # 5. Đánh giá chi tiết mô hình tốt nhất trên tập Validation
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    cm = confusion_matrix(all_targets, all_preds)
    print("\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT TRÊN TẬP VAL) ---")
    print(classification_report(all_targets, all_preds, target_names=unique_cats, digits=4))

    # 6. Vẽ biểu đồ 07: Đường cong huấn luyện (Training Curves)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    ep_range = range(1, epochs + 1)

    # Subplot 1: Loss
    axes[0].plot(ep_range, history['train_loss'], 'o-', color='#3b82f6', label='Train Loss', linewidth=2)
    axes[0].plot(ep_range, history['val_loss'], 's--', color='#ef4444', label='Val Loss', linewidth=2)
    axes[0].set_title('Đường cong Hàm mất mát (Cross-Entropy Loss)', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epoch', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=10, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(frameon=True, fontsize=10)

    # Subplot 2: Accuracy
    axes[1].plot(ep_range, history['train_acc'], 'o-', color='#10b981', label='Train Accuracy', linewidth=2)
    axes[1].plot(ep_range, history['val_acc'], 's--', color='#f59e0b', label='Val Accuracy', linewidth=2)
    axes[1].set_title('Đường cong Độ chính xác phân loại (%)', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epoch', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    axes[1].set_ylim(40, 102)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(frameon=True, fontsize=10)

    plt.suptitle(f"Tiến trình Huấn luyện Fine-tuned ResNet-18 (Best Val Acc: {best_val_acc:.1f}%)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig7_path = os.path.join(VIZ_DIR, "07_training_curves.png")
    plt.savefig(fig7_path)
    plt.close()
    print(f"  -> Đã lưu biểu đồ: {fig7_path}")

    # 7. Vẽ biểu đồ 10: Confusion Matrix phân loại trên tập Validation
    plt.figure(figsize=(9, 7), dpi=300)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.0f}%)"

    sns.heatmap(cm, annot=annot, fmt='', cmap='Greens', xticklabels=unique_cats, yticklabels=unique_cats, cbar=True)
    plt.title(f"Ma trận Nhầm lẫn Phân loại (Validation Set - Acc: {best_val_acc:.1f}%)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Loài dự đoán (Predicted)", fontsize=11, fontweight='bold')
    plt.ylabel("Loài thực tế (Ground Truth)", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig10_path = os.path.join(VIZ_DIR, "10_confusion_matrix_finetuned.png")
    plt.savefig(fig10_path)
    plt.close()
    print(f"  -> Đã lưu biểu đồ: {fig10_path}")

if __name__ == '__main__':
    main()
