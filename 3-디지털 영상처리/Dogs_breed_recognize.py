import cv2 as cv
import numpy as np
import tensorflow as tf
import winsound
import pickle
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

cnn = tf.keras.models.load_model('2024154004.h5')  # 모델 로드
dog_species = pickle.load(open('dog_species_names.txt', 'rb'))  # 견종 이름 리스트

class DogSpeciesRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('견종 인식 프로그램')
        self.setGeometry(200, 200, 800, 600)

        # 버튼 구성
        fileButton = QPushButton('강아지 사진 열기', self)
        recognitionButton = QPushButton('품종 인식', self)
        quitButton = QPushButton('나가기', self)

        fileButton.setGeometry(10, 10, 150, 30)
        recognitionButton.setGeometry(170, 10, 150, 30)
        quitButton.setGeometry(650, 10, 100, 30)

        fileButton.clicked.connect(self.pictureOpenFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        # QLabel로 이미지 표시
        self.label = QLabel(self)
        self.label.setGeometry(10, 50, 780, 540)
        self.label.setAlignment(Qt.AlignCenter)

    def pictureOpenFunction(self):
        fname, _ = QFileDialog.getOpenFileName(self, '강아지 사진 열기', '', 'Image files (*.jpg *.jpeg *.png)')
        if not fname:
            return
        self.img = cv.imread(fname)
        if self.img is None:
            QMessageBox.critical(self, '오류', '이미지를 불러올 수 없습니다.')
            return
        self.showImage(self.img)

    def recognitionFunction(self):
        if self.img is None:
            QMessageBox.warning(self, '경고', '먼저 이미지를 불러오세요.')
            return
        # 이미지 복사본 생성
        output_img = self.img.copy()
        x = np.reshape(cv.resize(self.img, (224, 224)), (1, 224, 224, 3))
        res = cnn.predict(x)[0]
        top5 = np.argsort(-res)[:5]
        top5_names = [dog_species[i] for i in top5]

        for i, idx in enumerate(top5):
            prob = f"{res[idx]*100:.2f}%"
            name = top5_names[i]
            text = f"{i+1}. {name}: {prob}"
            cv.putText(output_img, text, (10, 30 + i * 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.showImage(output_img)
        winsound.Beep(1000, 500)

    def showImage(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytesPerLine = ch * w
        qImg = QImage(rgb_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg).scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)

    def quitFunction(self):
        self.close()

# 실행부
app = QApplication(sys.argv)
win = DogSpeciesRecognition()
win.show()
app.exec_()
