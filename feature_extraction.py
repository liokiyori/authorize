from keras.preprocessing import image
from keras_facenet import FaceNet
import os
import numpy as np
import pandas as pd


class feature_extraction :
    def model(self):
        model = FaceNet()
        return model  # return the model here
        
    def feature_extraction(self, img_path):
        features = []
        labels = []
        model = self.model()
        for each in os.listdir(img_path):
            path = os.path.join(img_path, each)
            img = image.load_img(path)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            feature = model.embeddings(img)
            features.append(feature)
            labels.append(os.path.basename(img_path))
        return features, labels
    
    def transformation_dataframe(self,features, labels):
        dataframe = pd.DataFrame(np.array(features).reshape(-1,len(features)))
        dataframe = dataframe.T
        dataframe["label"] = labels
        return dataframe
    