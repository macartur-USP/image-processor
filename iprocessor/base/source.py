"""Basic class to download images from source."""
from pathlib import Path
from urllib import request


class Source():  # pylint: disable=too-few-public-methods
    """Base class to downlaod and store analyze images from source."""

    server = None
    store_path = None

    def __init__(self, server=None, store_path=None):
        """Source constructor receive the parameters below.

        Args:
            server(string): server name.
            store_path(string): path where the images will be stored.
        """
        self.server = server or ""
        self.store_path = (store_path or "/tmp/")

    def download_image(self, url, destination):
        """Download an image.

        Args:
            url(string): url where the image is available.
            destination(string): path where the image will be stored.
        """
        destination = Path(self.store_path + str(destination))
        request.urlretrieve(self.server+url, destination)
