import tensorflow as tf
import numpy as np
from PIL import Image

def prepare_image(filename: str, target_size=(28, 28)):
    # @note falta por um try/catch aqui

    # Carrega a imagem
    image = Image.open(filename)

    # redimensiona
    image = image.resize(target_size)

    # grayscale (apenas luminosidade)
    image = image.convert('L')
    # refaz terceira dimensão (ficaria dois eixos apenas sem esse comando)
    image = np.expand_dims(image, axis=-1)

    # dados da imagem (talvez precise arrumar caso seja RGBA)
    img_data = np.array(image)

    # casteia e normaliza
    img_data = img_data.astype('float32') / 255.0

    # tensor flow (parece opcional...)
    img_data = tf.convert_to_tensor(img_data)

    return np.expand_dims(img_data, axis=0)
