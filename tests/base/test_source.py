"""Module to test iprocessor.base.source."""
import unittest
from iprocessor.base.source import Source


class TestSource(unittest.TestCase):
    """Test Suite for iprocessor.base.source."""

    def setUp(self):
        """Create a source instance."""
        self.source = Source(server="https://vignette.wikia.nocookie.net")

    def test_create_source_without_parameters(self):
        """Test default source initialization."""
        source = Source()
        self.assertEqual("", source.server)
        self.assertEqual("/tmp/", source.store_path)

    def test_download_an_image(self):
        """Download a sample image from source created on setup."""
        url = "/joke-battles/images/a/ac/Tree.png/revision/latest?cb=20170827155628"
        destination = "sample_tree.png"
        self.source.download_image(url, destination)


if __name__ == '__main__':
    unittest.main()
