"""Module to test iprocessor.base.texture_processor."""
import os
import unittest
import numpy as np
from iprocessor.base.texture_processor import TextureProcessor
from skimage.feature import greycomatrix



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
