import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os

# Novos caminhos
TRAIN_PATH = "./datasets/treino"
VAL_PATH = "./datasets/validacao"
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 16

# Pré-processamento sem split (treino e validação já estão separados)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Geradores de dados
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Modelo CNN simples
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(train_generator, validation_data=val_generator, epochs=5)

# Salva o modelo
model.save("modelo_animais.h5")

# Função para prever nova imagem
def prever_imagem(caminho_imagem):
    imagem = load_img(caminho_imagem, target_size=(IMG_HEIGHT, IMG_WIDTH))
    imagem_array = img_to_array(imagem) / 255.0
    imagem_array = np.expand_dims(imagem_array, axis=0)
    previsao = model.predict(imagem_array)
    indice = np.argmax(previsao)
    classes = list(train_generator.class_indices.keys())
    print("Previsão:", classes[indice], f"({previsao[0][indice]*100:.2f}%)")
    plt.imshow(imagem)
    plt.title(f"Previsão: {classes[indice]}")
    plt.axis('off')
    plt.show()