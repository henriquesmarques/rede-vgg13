import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Verificar se a GPU está disponível
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Configuração do dataset CIFAR-10
cifar = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar.load_data()

# Normalização das imagens
train_images, test_images = train_images / 255.0, test_images / 255.0

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(train_images)

# Criação do modelo VGG13 com Dropout, Batch Normalization e Regularização L2
model = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.4),
    
    Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.4),
    
    Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.4),
    
    Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.4),
    
    Flatten(),
    Dense(4096, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(4096, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compilação do modelo
model.compile(
    optimizer=SGD(learning_rate=1e-2, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Treinando modelo com Data Augmentation
history = model.fit(train_images, train_labels, batch_size=32, epochs=20)

# Testando modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=64, verbose=1)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

# Lista de nomes das classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Prever as classes do conjunto de teste
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_labels.flatten()

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Calcular a acurácia por classe
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Imprimir a acurácia por classe
for i, acc in enumerate(class_accuracy):
    print(f'Acurácia da classe {class_names[i]}: {acc*100:.2f}%')
