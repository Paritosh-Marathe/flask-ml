from flask import Flask,jsonify,request,Blueprint
import json
import tensorflow as tf
from skimage import io

_model=''
classification_api= Blueprint('classification_api',__name__)
model=tf.keras.load_model(_model)

@classification_api.route('/classification')
def classification():
    sample=json.loads(request.data)['image']
    return jsonify(str(model.predict(sample)))
