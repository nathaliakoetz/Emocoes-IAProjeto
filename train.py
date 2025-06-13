
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import criar_modelo

# Caminhos das pastas
caminho_treino = 'train'
caminho_teste = 'test'

# Parâmetros
largura, altura = 48, 48
batch_size = 64
num_classes = 7

# Geradores de dados com normalização
datagen = ImageDataGenerator(rescale=1./255)

treino_generator = datagen.flow_from_directory(
    caminho_treino,
    target_size=(altura, largura),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=batch_size,
    shuffle=True
)

teste_generator = datagen.flow_from_directory(
    caminho_teste,
    target_size=(altura, largura),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=batch_size,
    shuffle=False
)

# Mostrar classes
print("Mapeamento das classes:", treino_generator.class_indices)

# Criar e treinar o modelo
model = criar_modelo()
model.fit(treino_generator, epochs=20, validation_data=teste_generator)

# Salvar modelo treinado
model.save('modelo_final.h5')
print("Modelo salvo como modelo_final.h5")
