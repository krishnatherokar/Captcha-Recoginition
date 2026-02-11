import numpy as np
import keras
from keras.ops import ctc_decode

HEIGHT = 50
WIDTH = 200

characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
num_classes = len(characters)
char_to_num = {char: i for i, char in enumerate(characters)}
num_to_char = {i: char for i, char in enumerate(characters)}

def process_images(img):
    img = img.convert("L")
    img = img.resize((WIDTH, HEIGHT))
    img = np.array(img) / 255.0
    return img

base_model = keras.saving.load_model("hf://krishnatherokar/captcha-recognition")

def predict_and_decode(img):
    # preprocess
    processed_image = process_images(img)
    test_input = np.expand_dims([processed_image], axis=-1)

    # predict
    preds = base_model.predict(test_input)

    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    decode = ctc_decode(
        preds,
        sequence_lengths=input_len,
        strategy='greedy'
    )[0][0]

    for result in decode:
        text = "".join([num_to_char[int(x)] for x in result if x >= 0 and x < num_classes])
        return text