"""Module to test iprocessor.base.processor."""
import os
import unittest
import numpy
from pathlib import Path
from iprocessor.base.processor import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    """Test Suite for iprocessor.base.source."""

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

    def test_resize_image(self):
        self.image_processor.resize_image()
        self.assertEqual(self.image_processor.image.shape,
                         self.image_processor.resize_shape)
        # verify last values of image
        self.assertEqual(82, self.image_processor.image[-1][-1][0])
        self.assertEqual(47, self.image_processor.image[-1][-1][1])
        self.assertEqual(255, self.image_processor.image[-1][-1][2])

    def test_convert_image_to_hsv(self):
        self.image_processor.resize_image()
        self.image_processor.convert_to_hsv()
        self.assertEqual(self.image_processor.image.shape, (320, 240, 3))

    def test_split_into_blocks(self):
        self.image_processor.resize_image()
        self.image_processor.convert_to_hsv()
        self.image_processor.split_into_blocks()
        self.assertEqual(len(self.image_processor.image_blocks), 352)

    def test_create_color_histogram(self):
        """Test whether the color histogram with tree feactures are created.

        The feactures are Hue, saturation and value.
        """

        self.image_processor.resize_image()
        self.image_processor.convert_to_hsv()
        self.image_processor.split_into_blocks()

        histograms = ImageProcessor.create_color_histograms(
            self.image_processor.image_blocks[0], 15)
        self.assertEqual(len(histograms), 3)

    def test_create_histogram(self):
        """Verify whether the some image block have the expected values.

        The histogram using the hue feacture is created and the x, y values are
        verified.
        """
        self.image_processor.resize_image()
        self.image_processor.convert_to_hsv()
        self.image_processor.split_into_blocks()
        image = self.image_processor.image_blocks[341]
        x,y = ImageProcessor.create_histogram(image, feacture=0)

        expected_x = [167, 6, 1, 7, 1, 10, 2, 2, 5, 0, 2, 8, 2, 3, 9]
        expected_y = [0.03287982, 0.09863946, 0.16439909, 0.23015873,
                      0.29591837, 0.361678, 0.42743764, 0.49319728, 0.55895692,
                      0.62471655, 0.69047619, 0.75623583, 0.82199546,
                      0.8877551, 0.95351474]
        self.assertTrue((x == expected_x).all())
        self.assertTrue(numpy.allclose(y, expected_y))

if __name__ == '__main__':
    unittest.main()
