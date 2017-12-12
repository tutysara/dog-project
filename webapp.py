import cv2
from glob import glob
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import regularizers
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
inceptionv3_model = InceptionV3(include_top=False, weights='imagenet')
model = Sequential()
#model.add(Flatten(input_shape=(5,5,2048)))
model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(133, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

### load weights from file
weight_file_name = "saved_models/weights.best.mymodel.inceptionv3.hdf5"
model.load_weights(weight_file_name)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def extract(tensor, model, preprocess_input):
    return model.predict(preprocess_input(tensor))

def predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract(path_to_tensor(img_path), inceptionv3_model, inception_v3_preprocess_input)
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def dog_detector(img_path):
    img =inception_v3_preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(inceptionv3_model.predict(img))
    return ((prediction <= 268) & (prediction >= 151))

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def detect_breed(img_path):
    is_dog = False
    is_human = False
    if dog_detector(img_path):
        is_dog = True
    elif face_detector(img_path):
        is_human = True
    else:
        print("No dog or humans found")
        return
    dog_breed = predict_breed(img_path)
    if(is_human):
        print("Hey there...")
        print("You look like a...\n {dog_breed}".format(dog_breed=dog_breed))
    else:
        print("This look like a...\n {dog_breed}".format(dog_breed=dog_breed))


#print(detect_breed("trisha.jpg"))

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/dog_detector/<string:url>', methods=['GET'])
def detect_dog_flask(url):
    res=dog_detector(url)
    return jsonify(res)

@app.route('/face_detector/<string:url>', methods=['GET'])
def detect_face_flask(url):
    res=face_detector(url)
    return jsonify(res)

@app.route('/predict_breed/<string:url>', methods=['GET'])
def detect_breed_flask(url):
    res=predict_breed(url)
    return jsonify(res)


app.run(debug=False)
