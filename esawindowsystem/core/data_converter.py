import pathlib
from pathlib import Path
from typing import Any

import cv2

import numpy as np
import numpy.typing as npt
from PIL import Image

from core.BCJR_decoder_functions import ppm_symbols_to_bit_array
from core.utils import tobits
from ppm_parameters import GREYSCALE

IMG_SUFFIXES: list[str] = [".png", ".jpg", ".jpeg"]


def _validate(user_data: Any, data_type: Any) -> None:
    """Validate the user data. Raise a type error if it is not valid. """
    if not isinstance(user_data, data_type):
        raise TypeError(f"Input data must be a string. Input data is a {type(user_data)}")


class DataConverter:
    def __init__(self, user_data: Any):
        self.bit_array: npt.NDArray[np.int_]
        match user_data:
            case str():
                self.bit_array = self.from_string(user_data)
            case pathlib.Path() if user_data.suffix in IMG_SUFFIXES:
                self.bit_array = self.from_image(user_data, greyscale=GREYSCALE)
            # CSV not yet implemented
            # case pathlib.Path() if user_data.suffix == '.csv':
            #     self.bit_array = []
            case _:
                raise TypeError('Data type not supported')

    def from_string(self, user_data: str) -> npt.NDArray[np.int_]:
        """Convert a string to a bit array. """
        _validate(user_data, str)

        return np.array(tobits(user_data))

    def from_image(self, filepath: Path, greyscale=True) -> npt.NDArray[np.int_]:
        """Take a filepath and convert the image to a bit stream. """
        _validate(filepath, Path)
        if filepath.suffix not in IMG_SUFFIXES:
            raise ValueError("File does not have the correct filetype. Should be one of .png, .jpg, or .jpeg")

        img_mode = "L" if greyscale else "1"

        # In the case of greyscale, each pixel has a value from 0 to 255.
        # This would be the same as saying that each pixel is a symbol, which should be mapped to an 8 bit sequence.
        if greyscale:
            img_arr = np.asarray(Image.open(filepath).convert(img_mode))
            bit_array = ppm_symbols_to_bit_array(img_arr.flatten(), 8)
        else:
            img = cv2.imread(str(filepath), 2)
            ret, bw_img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
            bw_img = np.asarray(bw_img/255, dtype=int)
            
            # converting to its binary form
            bw = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
            bit_array = bw_img.flatten()

        return bit_array

    def from_csv(self, filpath: Path):
        pass


def payload_to_bit_sequence(payload_type: str, **kwargs) -> npt.NDArray[np.int_]:
    """Convert an image or string to a bit sequence. """
    d: DataConverter

    match payload_type:
        case 'string':
            d = DataConverter("Hello World!")
            return d.bit_array
        case 'image':
            filepath = kwargs.get('filepath')
            if not filepath:
                raise ValueError("File path cannot be empty. ")
            file = Path(filepath)
            d = DataConverter(file)
            return d.bit_array
        case _:
            raise ValueError("Payload type not recognized. Should be one of 'string' or 'image'")
