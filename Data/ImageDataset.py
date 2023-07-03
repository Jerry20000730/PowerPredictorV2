from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch


def get_info(filename):
    """
    This function takes a filename as input, extracts information from it, and returns a tuple of
    three elements.

    :param filename: The input parameter is a string representing a filename
    :return: a tuple containing three elements: the first element is the first part of the filename
    (indicating the terrainID), the second element is the second part of the filename (indicating position parameter),
    and the third element is the third part of the filename (indicating direction parameter).
    """
    elements = filename.split('_')
    if elements[1] == '0' or elements[1] == '1' or elements[1] == '2' or elements[1] == '3':
        if elements[2] == '18':
            elements[2] = '4'
        elif elements[2] == '19':
            elements[2] = '5'
        elif elements[2] == '20':
            elements[2] = '6'
    return elements[0], elements[1], elements[2]


class Image2HeightmapDataset(Dataset):
    """
        This dataset is prepared for the first layer of training
        to train the relationship between an input image and its
        corresponding heightmap
        A heightmap means the relative height between the current position
        and height of the position mapped to each pixel of the image
        The heightmap has already been normalized before the training
        See: Preprocessing/Normalization.py
    """

    def __init__(self, image_src_folder_path, heightmap_src_folder_path, transform=None):
        self.image_src_folder_path = image_src_folder_path
        self.heightmap_src_folder_path = heightmap_src_folder_path
        self.transform = transform
        self.image_extension = ['.png', '.PNG', '.jpg', '.JPG']
        image_heightmap_link = []
        for dirname in os.listdir(image_src_folder_path):
            for filename in os.listdir(os.path.join(image_src_folder_path, dirname)):
                if any(filename.endswith(extension) for extension in self.image_extension):
                    terrainID, position, direction = get_info(filename)
                    # if the terrainID_position_direction_mat_normalized.npy exists
                    if os.path.exists(os.path.join(self.heightmap_src_folder_path,
                                                   "{}_{}_{}_mat_normalized.npy".format(terrainID, position,
                                                                                        direction))):
                        # if existed, append to the list of link
                        image_heightmap_link.append(
                            {
                                "img_path": os.path.abspath(
                                    os.path.join(self.image_src_folder_path, dirname, filename)),
                                "mat_path": os.path.abspath(os.path.join(self.heightmap_src_folder_path,
                                                                         "{}_{}_{}_mat_normalized.npy".format(terrainID,
                                                                                                              position,
                                                                                                              direction)))
                            }
                        )
                    else:
                        print("[ERROR] " + os.path.join(self.heightmap_src_folder_path, "{}_{}_{}_mat_normalized.npy"
                                                        .format(terrainID, position, direction)) + " does not exist")
        self.image_heightmap_link = image_heightmap_link

    def __len__(self):
        """
            This function returns the number of the images in the dataset

            :return: The length of the "images" attribute of the object.
        """
        return len(self.image_heightmap_link)

    def __getitem__(self, index):
        """
            This function returns an image and its corresponding power group from a list of images, with an
            optional transformation (see transforms tutorial: https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html)
            applied to the image.

            :param index: The index parameter is the index of the item that needs to be retrieved from the
            dataset. In this case, it is used to retrieve the filepath and power_group of the image at the
            specified index

            :return: A tuple containing the transformed image and the power group associated with the image
            at the given index.
        """
        image_path, heightmap_path = self.image_heightmap_link[index]['img_path'], self.image_heightmap_link[index]['mat_path']
        img = Image.open(image_path).convert('RGB')
        heightmap = np.load(heightmap_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, heightmap
