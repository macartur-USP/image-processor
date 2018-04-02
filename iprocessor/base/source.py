"""Basic class to download images from source."""
from pathlib import Path
from urllib import request
from urllib.parse import urlencode
from mapbox import Static

class ImageNet:

    def __init__(self):
        self.server = 'http://www.image-net.org'
        self.api_address = f'{self.server}/api/text/imagenet.synset.geturls'
        self.destination = '/tmp/imagenet'

    def download(self, identifier):
        """Download an image from Imagenet based on identifier.

        Args:
            identifier: string that identifier a set of images
        """
        url = f'{self.api_address}?wnid={identifier}'
        response = request.urlopen(url)
        image_urls = response.readlines()
        urls_error = []
        for url in image_urls:
            url = url.decode('utf-8').replace('\r','').replace('\n','')
            if (url == ''):
                continue
            file_name = url.split('/')[-1]
            destination = f'{self.destination}/{file_name}'
            try:
                response = request.urlretrieve(url, destination)
            except:
                urls_error.append(url)

class MapBox:
    def __init__(self, service, access_token):
        """Built a instance of mapbox.

        References: https://www.mapbox.com/api-documentation/#maps

        Args:
            service: The name of service that can be used as image source.
                     e.g. ['mapbox.satellite','mapbox.streets']
        """
        self.token = access_token or ''
        self.service = service or 'mapbox.satellite'
        self.source = Static(access_token=self.token)
        self.destination = f'/tmp/mapbox/{service.replace("mapbox","")}'

    def download(self, coordinate=('0','0'), params={}):
        """Download an image based on coordinate.

        Args:
         coordinate: (lat,lon): Latitude and Longitude coordinates.
         params: Dict with the following attributes.
            zoom
        """
        params['lat'] = coordinate[0]
        params['lon'] = coordinate[1]

        response = self.source.image(self.service, **params)

        if response.status_code != 200:
            raise "The coordinate can't be found."

        image = response.content
        destination = f'{self.destination}/{"_".join(coordinate)}.png'

        with open( destination, 'wb') as output:
            output.write(image)

class Google:

    def __init__(self, service, access_token):
        """Built a instance of google.

        Args:
            service: Type of service that will be requested.
                     Types: ['staticmap', 'streetview']
            access_token: token to handle the google api
        """
        self.server = 'https://maps.googleapis.com'
        self.service = service or 'staticmap'
        self.api_address = f'{self.server}/maps/api/{service}'
        self.token = access_token or ""
        self.destination = f'/tmp/google_maps/{service}'

    def download(self, coordinate=('0','0'), params={}):
        """Download an image based on coordinate.
        Args:
         coordinate: (lat,lon):
         params: Dict with the following attributes.
                 maptype
                 zoom
                 size
                 key
        """
        params['key'] = self.token
        if self.service == 'staticmap':
            params['center'] = f'{",".join(coordinate)}'

        if self.service == 'streetview':
            params['location'] = f'{",".join(coordinate)}'

        get_params = urlencode(params)
        url = f'{self.api_address}?{get_params}'
        destination = f'{self.destination}/{"_".join(coordinate)}.png'
        request.urlretrieve(url, destination)
