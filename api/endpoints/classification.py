from flask import Flask,jsonify,request,Blueprint
from model import ConvNet
import numpy as np
import tensorflow as tf
from skimage import io
from skimage.transform import resize
import json
#import tensorflow as tf
#from skimage import io
import pickle
model= ConvNet()
_model='models/model_weights'
classification_api = Blueprint('classification_api', __name__)
model.load_weights(_model)
#model=pickle.load(open(_model,'rb+'))

@classification_api.route('/classification')
def classification():
    sample=json.loads(request.data)['data']
    img=resize(io.imread(sample,as_gray=True),(28,28))
    predictions=model(img)
    ret_val= np.argmax(predictions.numpy())
    return str(ret_val)#sample#jsonify(str(model.predict(sample)))
