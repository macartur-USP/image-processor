"""Module to test iprocessor.base.processor."""
import os
import unittest
import numpy
from pathlib import Path
from iprocessor.base.processor import ImageProcessor, ColorProcessor
from skimage.exposure import histogram
from skimage import data, color


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
        """Test create histogram using first color feacture."""
        image = self.image[0:self.image.shape[0], 0: self.image.shape[1],0]
        ex, ey = histogram(image, nbins=15)
        x, y = ColorProcessor.create_histogram(self.image, feacture=0)
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



class TestImageProcessor(unittest.TestCase):
    """Test Suite for ImageProcessor class."""

    def setUp(self):
        self.sources = Path(os.path.dirname(__file__) + '/../sources/')
        self.image_processor = ImageProcessor()
        self.image_processor.open(self.sources /  'sample_tree.png')

    def test_set_a_valid_image(self):
        self.image_processor.open(self.sources /  'sample_tree.png')
        self.assertIsNotNone(self.image_processor.image)


    def test_set_image_with_none(self):
        self.image_processor.open(None)
        self.assertIsNone(self.image_processor.image)

        self.image_processor.open()
        self.assertIsNone(self.image_processor.image)

    def test_set_invalid_image(self):
        self.image_processor.open('/test/')
        self.assertIsNone(self.image_processor.image)

    def test_pre_process(self):
        self.image_processor.pre_process()

        self.assertEqual(self.image_processor.image.shape,
                         self.image_processor.resize_shape)

        last_pixel = self.image_processor.image[-1][-1]
        expected_pixel = [0.69471154, 0.81568627, 1.]
        self.assertTrue(numpy.allclose(last_pixel, expected_pixel))

    def test_split_in_blocks(self):
        self.image_processor.pre_process()
        image_blocks = self.image_processor.split_in_blocks()
        self.assertEqual(len(image_blocks), 352)

    def test_create_color_histogram(self):
        """Test whether the color histogram with tree feactures are created.

        The feactures are Hue, saturation and value.
        """
        self.image_processor.pre_process()
        image_blocks = self.image_processor.split_in_blocks()

        histograms = ColorProcessor.create_color_histograms(image_blocks[0])
        self.assertEqual(len(histograms), 3)

    def test_create_histogram(self):
        """Verify whether the some image block have the expected values.

        The histogram using the hue feacture is created and the x, y values are
        verified.
        """
        self.image_processor.pre_process()
        block = self.image_processor.split_in_blocks()[341]
        x,y = ColorProcessor.create_histogram(block, feacture=0)
        expected_x = [167, 6, 1, 7, 1, 10, 2, 2, 5, 0, 2, 8, 2, 3, 9]
        expected_y = [0.03287982, 0.09863946, 0.16439909, 0.23015873,
                      0.29591837, 0.361678, 0.42743764, 0.49319728, 0.55895692,
                      0.62471655, 0.69047619, 0.75623583, 0.82199546,
                      0.8877551, 0.95351474]
        self.assertTrue((x == expected_x).all())
        self.assertTrue(numpy.allclose(y, expected_y))

if __name__ == '__main__':
    unittest.main()
