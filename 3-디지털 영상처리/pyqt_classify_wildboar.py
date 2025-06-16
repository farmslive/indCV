import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# ===== CNN + LSTM 모델 정의 포함 =====
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

# ===== 클래스 이름 정의 (학습 순서와 동일해야 함) =====
class_names = ['cat', 'deer', 'dog', 'pig', 'wild_boar']  # 예시

# ===== 모델 로드 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_Classifier(num_classes=len(class_names))
model.load_state_dict(torch.load("model\cnn_lstm_wildboar.pth", map_location=device))
model.to(device)
model.eval()

# ===== 이미지 전처리 정의 =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== 예측 함수 =====
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()
    return class_names[pred_idx], confidence

# ===== PyQt GUI 클래스 =====
class ImageClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("동물 이미지 분류기")
        self.setGeometry(200, 200, 400, 500)
        self.setAcceptDrops(True)

        self.image_label = QLabel("이미지를 드래그 앤 드롭하세요", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa;")
        self.image_label.setFixedSize(350, 350)

        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px;")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        try:
            image = QImage(file_path)
            pixmap = QPixmap.fromImage(image.scaled(350, 350, Qt.KeepAspectRatio))
            self.image_label.setPixmap(pixmap)

            pred, conf = predict_image(file_path)
            self.result_label.setText(f"예측: {pred} (신뢰도: {conf:.2f})")
        except Exception as e:
            self.result_label.setText(f"오류: {str(e)}")

# ===== 앱 실행 =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierGUI()
    window.show()
    sys.exit(app.exec_())
