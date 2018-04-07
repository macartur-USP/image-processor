"""Module to test the ColorProcessor class."""
import numpy
import unittest
from skimage import data, color
from skimage.exposure import histogram
from iprocessor.color_processor import ColorProcessor


class TestColorProcessor(unittest.TestCase):
    """Test Suite for ColorProcessor class."""

    def setUp(self):
        """SetUp the tests."""
        self.image = data.astronaut()
        self.image_hsv = color.rgb2hsv(self.image)

    def test_convert_rgb_to_hsv(self):
        """Test convert rgb to hsv."""
        converted = ColorProcessor.convert_rgb_to_hsv(self.image)
        self.assertTrue(numpy.allclose(converted, self.image_hsv))

    def test_create_histogram(self):
        """Test create histogram using first color feature."""
        image = self.image[0:self.image.shape[0], 0: self.image.shape[1],0]
        ex, ey = histogram(image, nbins=15)
        x, y = ColorProcessor.create_histogram(self.image, feature=0)
        self.assertTrue((x == ex).all())
        self.assertTrue(numpy.allclose(y,ey))

    def test_create_color_histograms(self):
        """Test create color histogram for all hsv color space channels."""
        result = ColorProcessor.create_color_histograms(self.image)
        for n in range(3):
            image = self.image[0:self.image.shape[0], 0: self.image.shape[1],n]
            ex, ey = histogram(image, nbins=15)

            x = result[n][0]
            y = result[n][1]

            self.assertTrue((x == ex).all())
            self.assertTrue(numpy.allclose(y, ey))
