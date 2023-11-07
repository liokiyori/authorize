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
        for dirpath, dirnames, filenames in os.walk(img_path):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                img = image.load_img(path)
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                feature = model.embeddings(img)
                features.append(feature)
                labels.append(os.path.basename(dirpath))
        return features, labels
    
    def transformation_dataframe(self,features, labels):
        dataframe = pd.DataFrame(np.array(features).reshape(-1,len(features)))
        dataframe = dataframe.T
        dataframe["label"] = labels
        return dataframe
    
    def sauvegarde_csv(self,dataframe):
        dataframe.to_csv("data.csv", index=False)
        return dataframe