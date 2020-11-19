"""
Adapted from https://github.com/microsoft/CameraTraps/blob/master/detection/run_tf_detector.py
and https://github.com/microsoft/CameraTraps/blob/master/visualization/visualization_utils.py
"""

import glob
import math
import os
from io import BytesIO
from typing import Union

import numpy as np
import requests
from PIL import Image

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.gif', '.png']


def is_image_file(s):
    """
    Check a file's extension against a hard-coded set of image file extensions
    """
    ext = os.path.splitext(s)[1]
    return ext.lower() in IMAGE_EXTENSIONS


def find_image_files(strings):
    """
    Given a list of strings that are potentially image file names, look for strings
    that actually look like image file names (based on extension).
    """
    return [s for s in strings if is_image_file(s)]


def find_images(dir_name, recursive=False):
    """
    Find all files in a directory that look like image file names
    """
    if recursive:
        strings = glob.glob(os.path.join(dir_name, '**', '*.*'),
                            recursive=True)
    else:
        strings = glob.glob(os.path.join(dir_name, '*.*'))

    image_strings = find_image_files(strings)

    return image_strings


def truncate_float(x, precision=3):
    """
    Function for truncating a float scalar to the defined precision.
    For example: truncate_float(0.0003214884) --> 0.000321
    This function is primarily used to achieve a certain float representation
    before exporting to JSON

    Args:
    x         (float) Scalar to truncate
    precision (int)   The number of significant digits to preserve, should be
                      greater or equal 1
    """

    assert precision > 0

    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplicatipon with factor, flooring, and
        # division by factor
        return math.floor(x * factor) / factor


def open_image(input_file: Union[str, BytesIO]) -> Image.Image:
    """Opens an image in binary format using PIL.Image and converts to RGB mode.

    This operation is lazy; image will not be actually loaded until the first
    operation that needs to load it (for example, resizing), so file opening
    errors can show up later.

    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes

    Returns:
        an PIL image object in RGB mode
    """
    if (isinstance(input_file, str)
            and input_file.startswith(('http://', 'https://'))):
        try:
            response = requests.get(input_file)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print('Error opening image {}: {}'.format(input_file, str(e)))
            raise
    else:
        image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L'):
        raise AttributeError(
            f'Image {input_file} uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')
    return image


def load_image(input_file: Union[str, BytesIO]) -> Image.Image:
    """Loads the image at input_file as a PIL Image into memory.

    Image.open() used in open_image() is lazy and errors will occur downstream
    if not explicitly loaded.

    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes

    Returns: PIL.Image.Image, in RGB mode
    """
    image = open_image(input_file)
    image.load()
    return image


def chunk_list(ls, n):
    """Splits a list into n even chunks.

    Args
    - ls: list
    - n: int, # of chunks
    """
    for i in range(0, n):
        yield ls[i::n]
