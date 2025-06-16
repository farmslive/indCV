import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===== 모델 정의 =====
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # ----- Channel Attention -----
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn

        # ----- Spatial Attention -----
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_map, max_map], dim=1)))
        x = x * spatial_attn

        return x

class CNN_LSTM_CBAM(nn.Module):
    def __init__(self, num_classes=5, lstm_hidden=256, lstm_layers=1):
        super(CNN_LSTM_CBAM, self).__init__()

        # Pretrained ResNet18 CNN backbone
        cnn = models.resnet18(pretrained=True)
        self.cnn_feature_extractor = nn.Sequential(*list(cnn.children())[:-2])  # output: Bx512xHxW

        # CBAM after CNN
        self.cbam = CBAM(512)

        # Feature map pooling
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # output: Bx512x4x4

        # LSTM: sequence length = 16, input dim = 512
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True)

        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.cnn_feature_extractor(x)  # Bx512xHxW
        x = self.cbam(x)                   # CBAM: Bx512xHxW
        x = self.pool(x)                   # Bx512x4x4

        # Flatten spatial to sequence
        x = x.view(batch_size, 512, -1).permute(0, 2, 1)  # Bx16x512

        lstm_out, (hn, _) = self.lstm(x)
        out = self.classifier(hn[-1])  # Last hidden state → classifier
        return out

# ===== 데이터 전처리 및 증강 =====
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== 데이터셋 불러오기 (두 번 로드해서 transform 다르게 적용) =====
base_dataset = datasets.ImageFolder('dataset/')  # 클래스 정보만 참조용
num_classes = len(base_dataset.classes)
dataset_size = len(base_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

split = int(np.floor(0.2 * dataset_size))
train_idx, val_idx = indices[split:], indices[:split]

train_dataset = datasets.ImageFolder('dataset/', transform=train_transform)
val_dataset = datasets.ImageFolder('dataset/', transform=val_transform)

train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=16, shuffle=True)
val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=16, shuffle=False)

# ===== 학습 준비 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_CBAM(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===== 학습 루프 =====
train_accs = []
val_accs = []
train_losses = []
val_losses = []

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_idx)

    # ===== 검증 루프 =====
    model.eval()
    val_loss = 0.0
    val_correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / len(val_idx)

    # === 정확도 기록 ===
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # 평균 loss 저장
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    print(f"[Epoch {epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# ===== 성능 평가 =====
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=base_dataset.classes))

print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=base_dataset.classes, yticklabels=base_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Validation Confusion Matrix")
plt.show()

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

# ===== Precision-Recall Curve (Multiclass) =====

# 1. 라벨을 One-Hot 벡터로 변환 (예: [0,1,2] → [[1,0,0],[0,1,0],[0,0,1]])
all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

# 2. 모델 출력 확률값 계산 (softmax 적용)
all_probs = []
model.eval()
with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())

all_probs = np.array(all_probs)

# 3. 클래스별 PR Curve 시각화
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
    ap = average_precision_score(all_labels_bin[:, i], all_probs[:, i])
    plt.plot(recall, precision, label=f"{base_dataset.classes[i]} (AP={ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Validation)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Epoch 별 Loss 그래프 =====
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Epoch별 Accuracy 그래프 =====
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_accs, label='Train Accuracy')
plt.plot(range(1, EPOCHS + 1), val_accs, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# ===== 모델 저장 =====
# torch.save(model.state_dict(), "cnn_lstm_wildboar.pth")




# torch.save(model.state_dict(), "cnn_lstm_cbam_wildboar.pth")

