image-processor
###############

Overview
********

This is a project created to identify trees in an image from satellites.

Install
*******

To install this project you must download the project from github, install the
requirements then install the project.
You could follow the steps below to install the project.

.. code-block:: shell

  # download the project
  git clone https://github.com/macartur-USP/image-processor.git

  # move to inside project folder
  cd image-processor

  # install the requirements
  pip install -r requirements.txt

  # install the project
  python setup.py install


How to use
**********


Model Tranning
==============

.. code-block:: python

  # Import the classifier from image-processor project
  from iprocessor.classifier import Classifier

  # Make the classifier
  classifier = Classifier()

  # Assign each pixel of an image as tree
  classifier.seed('./raw/selected', True)

  # Assign each pixel of an image as not tree
  classifier.seed('./raw/not_selected', False)

  # Fit the adaboost classifier
  classifier.fit()


Predicting
==========

.. code-block:: python

  from iprocessor.processor import ImageProcessor

  image_processor = ImageProcessor()

  image_processor.open('./raw/sample_image.png')

  features = image_processor.extract_features(image_processor.image)

  result = classifier.predict(features, shape=image_processor.image.shape)

  image_processor.save(result, './raw/result_image.png')
