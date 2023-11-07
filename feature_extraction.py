from keras_facenet import FaceNet
from PIL import Image
import os
import numpy as np
import pandas as pd
from mtcnn import MTCNN


class feature_extraction :
    def model(self):
        model = FaceNet()
        return model  # return the model here
        
    def feature_extraction(self, img_path):
        features = []
        model = self.model()
        detector = MTCNN()
        for each in os.listdir(img_path):
            path = os.path.join(img_path, each)
            image = Image.open(path)
            image = np.asarray(image)
            results = detector.detect_faces(image)
            x1, y1, width, height = results['box']
            face = image[y1:y1+height, x1:x1+width]
            face = Image.fromarray(face)
            face = face.resize((160, 160))
            face = np.asarray(face)
            face = face.astype('float32')
            mean, std = face.mean(), face.std()
            face = (face - mean) / std
            face = np.expand_dims(face, axis=0)
            embedding = model.embeddings(face)
            features.append(embedding)
        return features
    
    def transformation_dataframe(self,features):
        dataframe = pd.DataFrame(np.array(features).reshape(-1,len(features)))
        dataframe = dataframe.T
        return dataframe
    