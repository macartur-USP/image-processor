"""Module to test iprocessor.base.processor."""
import os
import unittest
from pathlib import Path
from iprocessor.base.processor import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    """Test Suite for iprocessor.base.source."""

    def setUp(self):
        self.image_processor = ImageProcessor()
        self.sources = Path(os.path.dirname(__file__) + '/../sources/')

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
        self.image_processor.open(self.sources /  'sample_tree.png')
        self.image_processor.resize_image()
        self.assertEqual(self.image_processor.image.shape,
                         self.image_processor.resize_shape)


    def test_create_color_histogram(self):
        self.image_processor.open(self.sources /  'sample_tree.png')
        self.image_processor.resize_image()
        self.image_processor.split_into_blocks()

        histograma = self.image_processor.create_color_histogram(self.image_processor.image_blocks[0])
        self.assertEqual(self.image_processor.image_blocks[0].shape, [])


if __name__ == '__main__':
    unittest.main()
