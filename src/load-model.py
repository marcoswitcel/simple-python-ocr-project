import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Carrega um modelo para testar predições')
parser.add_argument('--model-filename', help='nome do modelo a ser carregado', default='model.h5')

# parseia argumentos
args = parser.parse_args()

# modelo a ser carregado
model_filename = args.model_filename

model = tf.keras.models.load_model(model_filename)

model.summary()

# todo João, fazer predições com o modelo
