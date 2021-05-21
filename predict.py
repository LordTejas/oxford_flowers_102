# python3

import os
import json
import argparse


import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np


import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


'''
Storing data in JSON format
'''
def store_json(data, output_name, indent=4, unicode=False):
    with open(output_name + '.json', 'w') as f:
        accounts_info_json = json.dumps(data, indent=indent, ensure_ascii=unicode)
        f.write(accounts_info_json)


'''
Predicting classes
'''
def process_image(image):
    image = image.squeeze()
    image = tf.image.resize(image, [224, 224]) / 255
    image = image.numpy()
    return image


def predict(image, model, class_names, top_k=5):
    image = np.asarray(Image.open(image))
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    
    top_values, top_indices = tf.math.top_k(predictions, top_k)
    top_classes = [class_names[str(value)] for value in top_indices.cpu().numpy()[0]]
    
    return top_values.numpy()[0], top_classes
    
    
if __name__ == '__main__':
    
    # Command Line Setup
    parser = argparse.ArgumentParser(description="My ML model!")
    
    parser.add_argument("image", type=str,
                       help="path of image")
    parser.add_argument("model", type=str,
                       help="path of saved model")
    
    parser.add_argument("-k", "--top_k", type=int,
                       help="Return the top K most likely classes.")
    parser.add_argument("-c", "--category_names", type=str,
                       help="Path to a JSON file mapping labels to flower names.")
    argc = parser.parse_args()
    
#     print(argc)
    
    
    # Image Path
    image_path = argc.image
    assert os.path.exists(argc.image), "The image path doesn't exists, please provide a valid path!"
    
    # Model path
    model_path = argc.model
    assert os.path.exists(argc.model), "The model path doesn't exists, please provide a valid path!"
    
    # classes selection
    top_k = 5
    if argc.top_k:
        top_k = argc.top_k
        if top_k <= 0:
            print("[INFO] Invalid number of classes entered, defaulting it to 5.")
            top_k = 5
    
    # json file name
    class_name_file = "category_data.json"
    if argc.category_names:
        class_name_file = argc.category_names
    
    assert class_name_file.endswith('.json'), "Invalid json file, please provide a valid name for the json file."
    
    with open(class_name_file, 'rb') as f:
        class_names = json.load(f)
#     print(class_names)
    
    '''
    Part 1: Loading Model and predicting classes
    '''
    
    print(f"[INFO] Loading the keras model...")
    
    # Defining custom keras layer
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(224, 224, 3))
    
    # Loading model with custom keras layer object
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

    print(f"[INFO] Classifying image...")
    values, classes = predict(image_path, model, class_names, top_k=top_k)
    
#     '''
#     Part 2: Exporting data as JSON file
#     '''
    
#     print(f"[INFO] Storing the JSON data to path -> {json_out}")
#     data = dict(classes, values)
#     store_json(data, json_out)
    
    print(f"[DONE] Successfuly completed predictions")
    
    print("\n\n[DATA] The predictions along with confidence ->\n")
    for label, value in zip(classes, values):
        print(f"{label}:   {value * 100} %")
