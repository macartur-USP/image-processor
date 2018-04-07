"""Module with classes to classify image and identify trees."""
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from iprocessor.processor import ImageProcessor
from PIL import Image
from glob import glob


class Classifier:

    def __init__(self, algorithm="SAMME"):
        decision_tree_classifier = DecisionTreeClassifier(max_depth=1)
        self.adaboost = AdaBoostClassifier(decision_tree_classifier,
                                           algorithm=algorithm,
                                           n_estimators=10**6)

    def convert_image(self, path):
        files = glob(path+'/*')

        for image in files:
            image_processor = ImageProcessor()
            image_processor.open(image)
            image_processor.pre_process()
            fill_image = image_processor.get_fill_image(image_processor.image)
            new_name = image.replace('raw', 'white_black')
            image_processor.save(fill_image, new_name)

    def seed(self, path):
        files = glob(path+'/*')
        for image in files:
            image_processor = ImageProcessor()
            image_processor.open(image)
            image_processor.pre_process()
            features = image_processor.extract_features(image_processor.image)
            fill_image = image_processor.get_fill_image(image_processor.image)
            result = fill_image.reshape(fill_image.shape[0]*fill_image.shape[1])
            self.fit(features, result)

    def verify(self, path):
        files = glob(path+'/*')
        for image in files:
            image_processor = ImageProcessor()
            image_processor.open(image)
            image_processor.pre_process()
            features = image_processor.extract_features(image_processor.image)
            fill_image = image_processor.get_fill_image(image_processor.image)

            expected = fill_image.reshape(fill_image.shape[0]*fill_image.shape[1])
            result = self.predict(features)
            result_analise = np.all((result == expected))
            print((result == expected))
            if result_analise == False:
                print(image)

    def fit(self, X, y):
        self.adaboost.fit(X, y)

    def predict(self,X):
        return self.adaboost.predict(X)
