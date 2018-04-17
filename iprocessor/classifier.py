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
                                           n_estimators=6000)
        self.X = list()
        self.y = list()

    def seed(self, path, expected=True):
        files = glob(path+'/*')
        for image in files:
            image_processor = ImageProcessor()
            image_processor.open(image)
            features = image_processor.extract_features(image_processor.image)
            self.X.append(features)

            if expected == False:
                self.y.append(np.zeros((features.shape[0],)))
            else:
                self.y.append(np.ones((features.shape[0],)))

    def fit(self):
        self.X = np.vstack(self.X)
        self.y = np.hstack(self.y)
        self.adaboost.fit(self.X,self.y )

    def predict(self, X, shape=None):
        result = self.adaboost.predict(X)
        if shape:
            result = result.reshape((shape[0], shape[1]))
        return result
