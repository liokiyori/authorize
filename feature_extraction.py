
import os
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,Input
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

class feature_extraction :
    def model(self):
        num_classes = 2
        img_size = 224
        image_input = Input(shape=(img_size, img_size, 3))
        model = VGG16(input_tensor=image_input, include_top=False, weights='imagenet')
        return model  # return the model here
        
    def feature_extraction(self,img_path):
        features = []
        model = self.model()
        for each in os.listdir(img_path):
            path = os.path.join(img_path,each)
            img = image.load_img(path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            feature = model.predict(img_data)
            features.append(feature)
        return features