import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===== 모델 정의 =====
class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, num_classes=5, lstm_hidden=256, lstm_layers=1):
        super(CNN_LSTM_Classifier, self).__init__()
        cnn = models.resnet18(pretrained=True)
        self.cnn_feature_extractor = nn.Sequential(*list(cnn.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn_feature_extractor(x)
        x = self.pool(x)
        x = x.view(batch_size, 512, -1).permute(0, 2, 1)
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.classifier(hn[-1])
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
model = CNN_LSTM_Classifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===== 학습 루프 =====
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

# ===== 모델 저장 =====
# torch.save(model.state_dict(), "cnn_lstm_wildboar.pth")
