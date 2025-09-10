import tensorflow as tf
import numpy as np
import argparse

from input import prepare_image

parser = argparse.ArgumentParser(description='Carrega um modelo para testar predições')
parser.add_argument('--model-filename', help='nome do modelo a ser carregado', default='model.keras')
parser.add_argument('--digit-filename', help='nome/caminho da imagem a ser usada no teste', default='')

# parseia argumentos
args = parser.parse_args()

# modelo a ser carregado
model_filename = args.model_filename
digit_filename = args.digit_filename

model = tf.keras.models.load_model(model_filename)

model.summary()

# todo João, fazer predições com o modelo

# gerando dados por hora, mas o formato seria esse
image = np.random.rand(28, 28, 1).astype('float32')

# remodelando para incluir a dimensão extra do batch
data = np.expand_dims(image, axis=0)

if len(digit_filename) != 0:
    data = prepare_image(digit_filename)

# faz predição e retorna uma lista de probabilidades
predictions = model.predict(data)

print(f"Predições: {predictions}")

# Achando o índice da maior probabilidade
predicted_index = np.argmax(predictions)
# menos 1 porque acredito que a ordem dos labels seria : 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
print(f"Classe prevista: {predicted_index}")
