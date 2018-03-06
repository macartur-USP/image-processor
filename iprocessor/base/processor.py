"""Module to process images."""
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import exposure

class ImageProcessor:

    image = None
    path = None
    resize_shape = (320, 240, 4)
    image_blocks = []
    feactures = []

    def open(self, image_path=None):
        """Open an image to be processed.
        Args:
            image_path: path of an image
        """
        if image_path is None:
            return

        try:
            self.image = imread(image_path)
            self.path = image_path
        except FileNotFoundError:
            self.image = None
            print(f'image {image_path} not found !!!')

    def pre_process(self):
        """Preprocess the image setted.

        This method will:
        - resize the image to 320 x 240 pixels
        - update image_blocks with 15x15 pixels of the current image
        - extract feactures from each block
        """
        self.resize_image()
        self.split_into_blocks()


    def resize_image(self):
        """Resize the current image."""
        self.image = resize(self.image, self.resize_shape, mode='reflect')

    def split_into_blocks(self):
        """Split current image into 15x15 pixels blocks."""
        for line in range(0,320, 15):
            for column in range(0, 240, 15):
                self.image_blocks.append(self.image[line:line+15,column:column+15])

    def extract_feactures(self):
        """"""
        pass

    def create_color_histogram(self, image, nbins=15):
        return exposure.histogram(image, nbins=nbins)


    def save_image(self, image,path):
        imsave(path, image)
