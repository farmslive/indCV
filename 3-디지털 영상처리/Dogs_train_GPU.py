import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import pathlib
import pickle

# =======================
# ✅ GPU 사용 가능 여부 확인
# =======================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 자동 할당 제한 해제 (선택적)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 사용 중: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("❌ GPU를 사용할 수 없습니다. CPU로 실행됩니다.")

# =======================
# ✅ 데이터 로드
# =======================
data_path = pathlib.Path('datasets/stanford_dogs/images/images')

train_ds = image_dataset_from_directory(
    data_path, validation_split=0.2, subset='training',
    seed=123, image_size=(224, 224), batch_size=16)

test_ds = image_dataset_from_directory(
    data_path, validation_split=0.2, subset='validation',
    seed=123, image_size=(224, 224), batch_size=16)

# =======================
# ✅ 모델 정의 (DenseNet121 기반)
# =======================
base_model = DenseNet121(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))

cnn = Sequential([
    Rescaling(1.0 / 255.0),
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.75),
    Dense(units=120, activation='softmax')
])

# =======================
# ✅ 컴파일 및 학습
# =======================
cnn.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-6),
    metrics=['accuracy']
)

hist = cnn.fit(train_ds, epochs=20, validation_data=test_ds, verbose=2)

# =======================
# ✅ 정확도 출력 및 저장
# =======================
print('정확률 =', cnn.evaluate(test_ds, verbose=0)[1] * 100)
cnn.save('2024154004.h5')

# 클래스 이름 저장
with open('dog_species_names.txt', 'wb') as f:
    pickle.dump(train_ds.class_names, f)

# =======================
# ✅ 그래프 시각화
# =======================
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()
