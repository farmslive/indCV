import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget
from PyQt5.QtGui import QPainter, QImage, QPen, QPixmap
from PyQt5.QtCore import Qt, QPoint
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# 모델 로드
model = load_model("cnn_mnist_model.h5")

class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.last_point = QPoint()
        self.drawing = False

    def paintEvent(self, event):
        canvas = QPainter(self)
        canvas.drawImage(0, 0, self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.LeftButton):
            painter = QPainter(self.image)
            pen = QPen(Qt.white, 18, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def clear(self):
        self.image.fill(Qt.black)
        self.update()

    def getImageForPrediction(self):
        buffer = self.image.bits().asstring(self.image.width() * self.image.height() * 4)
        img = Image.frombytes("RGBA", (280, 280), buffer)
        img = img.convert("L").resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 필기 숫자 인식")
        self.setFixedSize(320, 400)

        self.drawing_widget = DrawingWidget()
        self.drawing_widget.setParent(self)
        self.drawing_widget.move(20, 20)

        self.result_label = QLabel("예측: ", self)
        self.result_label.setGeometry(20, 310, 280, 30)

        self.predict_btn = QPushButton("예측", self)
        self.predict_btn.setGeometry(20, 350, 60, 30)
        self.predict_btn.clicked.connect(self.predict)

        self.clear_btn = QPushButton("지우기", self)
        self.clear_btn.setGeometry(100, 350, 60, 30)
        self.clear_btn.clicked.connect(self.drawing_widget.clear)

    def predict(self):
        img_array = self.drawing_widget.getImageForPrediction()
        prediction = np.argmax(model.predict(img_array, verbose=0))
        self.result_label.setText(f"예측: {prediction}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
