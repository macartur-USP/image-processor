"""Module to test iprocessor.base.texture_processor."""
import os
import unittest
import numpy as np
from iprocessor.base.texture_processor import TextureProcessor
from skimage.feature import greycomatrix
from skimage import data


class TestImageProcessor(unittest.TestCase):
    """Test Suite for ImageProcessor class."""

    def test_coocorrence_matrix_angle_0(self):
        """Test whether the coocorrence matrix with angle 0 is right. """

        image = np.array([[1,1,5,6,8],
                          [2,3,5,7,1],
                          [4,5,7,1,2],
                          [8,5,1,2,5]], dtype=np.uint8)
        result = TextureProcessor.coocorrence_matrix(image, levels=9)

        expected_angle_0 = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 2, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 1, 2, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 2, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0]]
        self.assertTrue((result[:,:, 0, 0] == expected_angle_0).all())

    def test_local_binary_pattern(self):
        """Test whether the LBP method."""
        image = np.array([[1,1,5,6,8],
                          [2,3,5,7,1],
                          [4,5,7,1,2],
                          [8,5,1,2,5]], dtype=np.uint8)
        result = TextureProcessor.local_binary_pattern(image)
        expected_result = [[193., 243., 193., 65.,  0.],
                           [193., 227., 103., 0., 126.],
                           [193., 99., 0., 255., 104.],
                           [0., 30., 63., 11., 0.]]
        self.assertTrue(np.allclose(result, expected_result))

    def test_gabor_filter(self):
        """Test Texture gabor filter method."""
        image = np.array([[1,1,5,6,8],
                          [2,3,5,7,1],
                          [4,5,7,1,2],
                          [8,5,1,2,5]], dtype=np.uint8)
        real, img = TextureProcessor.gabor_filter(image, frequency=1, theta=0)

        expected_real = [[1, 1, 4, 6, 6],
                         [2, 3, 5, 5, 2],
                         [4, 4, 5, 2, 2],
                         [7, 4, 2, 2, 4]]

        expected_imaginary = [[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]]

        self.assertTrue((real == expected_real).all())
        self.assertTrue((img == expected_imaginary).all())

    def test_entropy(self):
        """Test Texture entropy method."""
        image = np.array([[1,1,5,6,8],
                          [2,3,5,7,1],
                          [4,5,7,1,2],
                          [8,5,1,2,5]], dtype=np.uint8)

        entropy = TextureProcessor.entropy(image)
        expected_entropy = [[2.72321967, 2.72321967, 2.72321967, 2.72321967, 2.72321967],
                            [2.72321967, 2.72321967, 2.72321967, 2.72321967, 2.72321967],
                            [2.72321967, 2.72321967, 2.72321967, 2.72321967, 2.72321967],
                            [2.72321967, 2.72321967, 2.72321967, 2.72321967, 2.72321967]]

        self.assertTrue(np.allclose(entropy,expected_entropy))

    def test_get_texture_features(self):
        """Test get texture features"""
        image = np.array([[1,1,5,6,8],
                          [2,3,5,7,1],
                          [4,5,7,1,2],
                          [8,5,1,2,5]], dtype=np.uint8)
        features = TextureProcessor.get_texture_features(image)
        self.assertEqual(features.shape, (4,226))

    def test_get_texture_features_with_15_x_15_image(self):
        """Test get texture features with image with 15x15 pixels."""
        image = np.array([[1,1,5,6,8,1,1,5,6,8,1,1,5,6,8],
                          [2,3,5,7,1,2,3,5,7,1,2,3,5,7,1],
                          [4,5,7,1,2,4,5,7,1,2,4,5,7,1,2],
                          [8,5,1,2,5,8,5,1,2,5,8,5,1,2,5],
                          [1,1,5,6,8,1,1,5,6,8,1,1,5,6,8],
                          [2,3,5,7,1,2,3,5,7,1,2,3,5,7,1],
                          [4,5,7,1,2,4,5,7,1,2,4,5,7,1,2],
                          [8,5,1,2,5,8,5,1,2,5,8,5,1,2,5],
                          [1,1,5,6,8,1,1,5,6,8,1,1,5,6,8],
                          [2,3,5,7,1,2,3,5,7,1,2,3,5,7,1],
                          [4,5,7,1,2,4,5,7,1,2,4,5,7,1,2],
                          [8,5,1,2,5,8,5,1,2,5,8,5,1,2,5],
                          [1,1,5,6,8,1,1,5,6,8,1,1,5,6,8],
                          [2,3,5,7,1,2,3,5,7,1,2,3,5,7,1],
                          [4,5,7,1,2,4,5,7,1,2,4,5,7,1,2]], dtype=np.uint8)
        features = TextureProcessor.get_texture_features(image)
        self.assertEqual(features.shape, (15,606))
