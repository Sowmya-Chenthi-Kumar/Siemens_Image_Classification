import os
import shutil

from PIL import Image, ImageEnhance

import numpy as np


class ImagePreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def format_file_structure(folder_path):
        """
        A static method to move files from multiple subdirectories into one directory

        :param folder_path: Path to raw directory
        """
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        for sub in subfolders:
            for f in os.listdir(sub):
                src = os.path.join(sub, f)
                dst = os.path.join(folder_path, f)
                shutil.move(src, dst)

    @staticmethod
    def label_file(folder_path, label, extension='bmp'):
        """
        A static method to label files in a directory

        :param folder_path: Path to image directory
        :param label: str; Label
        :param extension: Extension of images in the directory (Defaults to bmp)
        """

        files = os.listdir(folder_path)
        for index, file in enumerate(files):
            os.rename(os.path.join(folder_path, file),
                      os.path.join(folder_path, ''.join([label, '.', str(index), '.', extension])))

    @staticmethod
    def process_image(image, assembly_line, size=(224, 224)):

        """

        A static method to preprocess file depending on the assembly line

        :param image: PIL image
        :param assembly_line: Assembly line
        :param size: Size of the input image (Defaults to 224 x 224 when using transfer learning algorithms)
        :return: Processed numpy image
        """

        # A contarst factor >2 leads to overexposure for Line 2
        if assembly_line == 2:
            contrast_factor = 2

        # A contarst factor >1.2 leads to overexposure for Line 4
        if assembly_line == 4:
            contrast_factor = 1

        img = image.resize(size, Image.ANTIALIAS)
        contrast = ImageEnhance.Contrast(img)
        processed_image = contrast.enhance(contrast_factor)
        x = np.array(processed_image)
        x = x.reshape(-1, size[0], size[1], 3)
        return x
