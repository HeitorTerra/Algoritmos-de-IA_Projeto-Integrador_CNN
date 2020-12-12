#%% Importando bibliotecas
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

#%% Carregando o dataset
# Observe que é  necessário colocar o diretório das imagens
# NÃO ESQUEÇA DE COLOCAR A SEGUNDA BARRA NO DIRETÓRIO
# Separando o dataset em treino e teste
#A base é a pasta onde se encontra as pastas teste e treino
base = 'C:\\Users\\theus\\Downloads\\cats_and_dogs_filtered\\cats_and_dogs_filtered'
#Abra a pasta treino copie o diretório e cole aqui
treino = 'C:\\Users\\theus\\Downloads\\cats_and_dogs_filtered\\cats_and_dogs_filtered\\train'
#Abra a pasta teste copie o diretório e cole aqui
teste = 'C:\\Users\\theus\\Downloads\\cats_and_dogs_filtered\\cats_and_dogs_filtered\\validation'

# Separando as imagens de gatos das imagens de cachorro
#Abra a pasta treino, após isso abra a pasta cats e cole aqui
treino_gatos_img = ['C:\\Users\\theus\\Downloads\\cats_and_dogs_filtered\\cats_and_dogs_filtered\\train\\cats{}'.format(i) for i in os.listdir(treino) if 'cat' in i] 
#Abra a pasta treino, após isso abra a pasta dogs e cole aqui
treino_cachorros_img = ['C:\\Users\theus\\Downloads\\cats_and_dogs_filtered\\cats_and_dogs_filtered\\train\\dogs{}'.format(i) for i in os.listdir(treino) if 'dog' in i]
#Abra a pasta teste, após isso abra a pasta dogs e cole aqui
teste_gatos_img = ['C:\\Users\\theus\\Downloads\\cats_and_dogs_filtered\\cats_and_dogs_filtered\\validation\\cats{}'.format(i) for i in os.listdir(teste) if 'cat' in i] 
#Abra a pasta teste, após isso abra a pasta dogs e cole aqui
teste_cachorros_img =  ['C:\\Users\\theus\\Downloads\\cats_and_dogs_filtered\\cats_and_dogs_filtered\\validation\\dogs{}'.format(i) for i in os.listdir(teste) if 'dog' in i] 

#%% Analisando o dataset
#Aqui foi colocado quantas imagens há de gatos e cachorros para
#treino, colocados manualmente.
num_gatos_treino = 1000
num_cachorros_treino = 1000
#Aqui foi colocado quantas imagens há de gatos e cachorros para
#teste, colocados manualmente.
num_gatos_teste = 500
num_cachorros_teste = 500
#Somando a quantidade total de teste e treino 
total_treino = num_gatos_treino + num_cachorros_treino
total_teste = num_gatos_teste + num_cachorros_teste

#Alguns prints para controle onde foram analizados os numeros de gato de teste e treino, número de cachorro de teste e treino,
#e por fim o total de teste e treino com estes somados
print('Número de gatos no dataset de treino:', num_gatos_treino)
print('Número de cachorros no dataset de treino:', num_cachorros_treino)

print('Número de gatos no dataset de teste:', num_gatos_teste)
print('Número de cachorros no dataset de teste:', num_cachorros_teste)
print("--")
print("Total dataset de treino:", total_treino)
print("Total dataset de teste:", total_teste)

#%% Configurando os parâmetros de treino
BATCH_SIZE = 100 # Quantidade de data que vai ser treinada por época
TAM_IMG  = 150 # Número de pixels das imagens que serão treinadas. Todas as imagens devem ter o mesmo tamanho, 150x150.

#%% Ampliando o dataset de treino

"""Para evitar os fenômenos de overfitting e underfitting, que costumam ocorrer em datasets pequenos, nós podemos ampliar nosso dataset, 
variando a visualização de uma mesma imagem. Como, por exemplo, rotacionando ela, ou aplicando zoom."""

# Configurando a transformação das imagens
image_gen_treino = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Transformando as imagens do dataset de treino
treino_data_gen = image_gen_treino.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=treino,
                                                     shuffle=True,
                                                     target_size=(TAM_IMG,TAM_IMG),
                                                     class_mode='binary')

# Plotando as imagens trasnformadas
augmented_images = [treino_data_gen[0][0][0] for i in range(5)]
fig, axes = plt.subplots(1, 5, figsize=(20,20))
axes = axes.flatten()
for img, ax in zip(augmented_images, axes):
    ax.imshow(img)
plt.tight_layout()
plt.show()

#%% Configurando o dataset de teste
image_gen_teste = ImageDataGenerator(rescale=1./255)

teste_data_gen = image_gen_teste.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=teste,
                                                 target_size=(TAM_IMG, TAM_IMG),
                                                 class_mode='binary')

#%% Criando o modelo do algoritmo
model = tf.keras.models.Sequential([
    #CNN
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(TAM_IMG, TAM_IMG, 3)), # Camada de convolução, com 32 filtros 3x3 e função de ativação relu
    tf.keras.layers.MaxPooling2D(2, 2), # Camada de MaxPooling 2x2

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    #ANN
    tf.keras.layers.Dropout(0.5), # Camada com dropout de 0.5, metade dos neurônios serão desativados aleatoriamente
    tf.keras.layers.Flatten(), # Camada que transforma as matrizes em vetores
    tf.keras.layers.Dense(512, activation='relu'), # Camada com 512 neurônios totalmente conectados e com ativação relu
    tf.keras.layers.Dense(2)
])

#%% Compilando o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Sumário do modelo
model.summary()

#%% Treinando o modelo
epochs=300 # Número de épocas que o modelo vai treinar

# Treinando
history = model.fit_generator(
    treino_data_gen,
    steps_per_epoch=int(np.ceil(total_treino / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=teste_data_gen,
    validation_steps=int(np.ceil(total_teste / float(BATCH_SIZE)))
)

#%% Plotando os resultados
acc = history.history['accuracy']
teste_acc = history.history['val_accuracy']

erro = history.history['loss']
teste_erro = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precisão do treino')
plt.plot(epochs_range, teste_acc, label='Precisão do teste')
plt.legend(loc='lower right')
plt.title('Precisões de treino e teste')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, erro, label='Erro do treino')
plt.plot(epochs_range, teste_erro, label='Erro do teste')
plt.legend(loc='upper right')
plt.title('Erros de treino e teste')
plt.show()