#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request

from PIL import Image



interpreter = tflite.Interpreter(model_path='model_2024_hairstyle_v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


classes = [
    'straight',
    'curly'
]

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# url = 'http://bit.ly/mlbookcamp-pants'
def preprocess_image(img):
    X = np.array(img, dtype=np.float32)
    X = np.expand_dims(X, axis=0)  # add batch dimension
    X = X / 255.0  # normalize
    return X

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))
    X = preprocess_image(img)
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    
    preds = interpreter.get_tensor(output_index)
    return float(preds[0][0])


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result


