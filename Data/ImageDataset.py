from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os


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


def query_power(df, terrainID, position, direction):
    """
    This function takes a dataframe, terrainID, position, and direction as inputs and returns the
    average power from the dataframe.

    :param df: a pandas DataFrame containing power data for different terrains, positions, and
    directions
    :param terrainID: The ID of the terrain being queried for power data
    :param position: The position parameter is an integer value representing the position of the car.
    :param direction: The direction parameter is an integer that represents the direction of the car.

    :return: a float value of the 'average_power' column from the input dataframe 'df' based on the
    conditions specified in the query. The conditions are that the 'terrainID' column should match
    the input 'terrainID', the 'position' column should match the input 'position' (converted to an
    integer), and the 'direction' column should match the input 'direction'.
    """

    # prevent future warning
    ser = df.loc[(df['terrainID'] == terrainID) & (df['position'] == int(position)) & (
            df['direction'] == int(direction)), 'average_power']
    return float(ser.iloc[0])


def query_power_class(df, terrainID, position, direction):
    """
        This function takes a dataframe, terrainID, position, and direction as inputs and returns the
        average power from the dataframe.

        :param df: a pandas DataFrame containing power data for different terrains, positions, and
        directions
        :param terrainID: The ID of the terrain being queried for power data
        :param position: The position parameter is an integer value representing the position of the car.
        :param direction: The direction parameter is an integer that represents the direction of the car.

        :return: a int value representing the class of the power
    """

    # prevent future warning
    ser = df.loc[(df['terrainID'] == terrainID) & (df['position'] == int(position)) & (
            df['direction'] == int(direction)), 'power_group']
    return int(ser.iloc[0])


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

    def __init__(self,
                 image_src_folder_path,
                 heightmap_src_folder_path,
                 transform=None):
        """

        :param image_src_folder_path: the folder for images
        :param heightmap_src_folder_path: the folder for **normalized** heightmap
        :param transform: transforms applied to the images
        """

        self.image_src_folder_path = image_src_folder_path
        self.heightmap_src_folder_path = heightmap_src_folder_path
        self.transform = transform
        self.image_extension = ['.png', '.PNG', '.jpg', '.JPG']
        self.image_heightmap_link = []
        for dirname in os.listdir(self.image_src_folder_path):
            for filename in os.listdir(os.path.join(self.image_src_folder_path, dirname)):
                if any(filename.endswith(extension) for extension in self.image_extension):
                    terrainID, position, direction = get_info(filename)
                    # if the terrainID_position_direction_mat_normalized.npy exists
                    if os.path.exists(os.path.join(self.heightmap_src_folder_path,
                                                   "{}_{}_{}_mat_normalized.npy".format(terrainID, position,
                                                                                        direction))):
                        # if existed, append to the list of link
                        self.image_heightmap_link.append(
                            {
                                "img_path": os.path.abspath(os.path.join(
                                    self.image_src_folder_path,
                                    dirname,
                                    filename
                                )),
                                "mat_path": os.path.abspath(os.path.join(
                                    self.heightmap_src_folder_path,
                                    "{}_{}_{}_mat_normalized.npy".format(terrainID,
                                                                         position,
                                                                         direction)
                                ))
                            }
                        )
                    else:
                        print("[ERROR] " + os.path.join(self.heightmap_src_folder_path, "{}_{}_{}_mat_normalized.npy"
                                                        .format(terrainID, position, direction)) + " does not exist")

    def __len__(self):
        """
            This function returns the number of the images in the dataset

            :return: The length of the "images" attribute of the object.
        """
        return len(self.image_heightmap_link)

    def __getitem__(self, index):
        """
            This function returns an image and its corresponding normalized heightmap from a list of images, with an
            optional transformation (see transforms tutorial: https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html)
            applied to the image.

            :param index: The index parameter is the index of the item that needs to be retrieved from the
            dataset. In this case, it is used to retrieve the filepath and power_group of the image at the
            specified index

            :return: A dict containing the transformed image and the normalized heightmap indicating the image
            at the given index.
        """
        image_path, heightmap_path = self.image_heightmap_link[index]['img_path'], self.image_heightmap_link[index][
            'mat_path']
        img = Image.open(image_path).convert('RGB')
        heightmap = np.load(heightmap_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, heightmap


class Heightmap2PowerclassDataset(Dataset):
    """
            This dataset is prepared for the second layer of training
            to train the relationship between an inferred normalized heightmap
            and the power classification

            The power classification is done during the preprocessing
            The classification is based on the distribution (histogram)
            of average power recorded

            A heightmap means the relative height between the current position
            and height of the position mapped to each pixel of the image
            The heightmap has already been normalized before the training
            See: Preprocessing/Normalization.py
        """

    def __init__(self,
                 img_src_folder_path,
                 heightmap_src_folder_path,
                 classification_datafile_src_path,
                 transform=None):
        """

        :param img_src_folder_path: the folder for images
        :param heightmap_src_folder_path: the **normalized** heightmap folder, by default,
        it is located at the Dataset/PowerCollection/label_heightmap_normalized
        :param classification_datafile_src_path: the power classification csv file
        """

        self.img_src_folder_path = img_src_folder_path
        self.heightmap_src_folder_path = heightmap_src_folder_path
        self.classification_datafile_src_path = classification_datafile_src_path
        self.image_extension = ['.png', '.PNG', '.jpg', '.JPG']
        self.transform = transform
        self.heightmap_classification_link = []
        self.pwdf = pd.read_csv(self.classification_datafile_src_path, dtype={'terrainID': str})
        for dirname in os.listdir(self.img_src_folder_path):
            for filename in os.listdir(os.path.join(img_src_folder_path, dirname)):
                if any(filename.endswith(extension) for extension in self.image_extension):
                    terrainID, position, direction = get_info(filename)
                    # if the terrainID_position_direction_mat_normalized.npy exists
                    if os.path.exists(os.path.join(self.heightmap_src_folder_path,
                                                   "{}_{}_{}_mat_normalized.npy".format(terrainID, position,
                                                                                        direction))):
                        power_grp = query_power_class(self.pwdf, terrainID, position, direction)
                        if power_grp is None:
                            print("[ERROR] The power group for [terrainID: {}, position: {}, direction: {}] does not "
                                  "exist")
                        self.heightmap_classification_link.append(
                            {
                                "heightmap_path": os.path.abspath(os.path.join(
                                    self.heightmap_src_folder_path,
                                    "{}_{}_{}_mat_normalized.npy".format(terrainID, position,
                                                                         direction)
                                )),
                                "power_group": power_grp
                            }
                        )
                    else:
                        print("[ERROR] " + os.path.join(self.heightmap_src_folder_path, "{}_{}_{}_mat_normalized.npy"
                                                        .format(terrainID, position, direction)) + " does not exist")

    def __len__(self):
        """
            This function returns the number of the images in the dataset

            :return: The length of the "images" attribute of the object.
        """
        return len(self.heightmap_classification_link)

    def __getitem__(self, index):
        """
            This function returns a normalized and its corresponding power group

            :param index: The index parameter is the index of the item that needs to be retrieved from the
            dataset. In this case, it is used to retrieve the filepath and power_group of the image at the
            specified index

            :return: A dict containing the normalized heightmap and the power group associated with the
            heightmap at the given index.
        """
        heightmap_path, power_grp = self.heightmap_classification_link[index]['heightmap_path'], \
        self.heightmap_classification_link[index]['power_group']
        heightmap = np.load(heightmap_path)
        if self.transform is not None:
            heightmap = self.transform(heightmap)
        return heightmap, power_grp
    

class Image2PowerclassDataset(Dataset):
    """
        This dataset is for training the inference directly between image and power class

        The power classification is done during the preprocessing
        The classification is based on the distribution (histogram)
        of average power recorded
    """

    def __init__(self,
                 img_src_folder_path,
                 classification_datafile_src_path,
                 transform=None):
        """

        :param img_src_folder_path: the folder for images
        :param classification_datafile_src_path: the power classification csv file
        """

        self.img_src_folder_path = img_src_folder_path
        self.classification_datafile_src_path = classification_datafile_src_path
        self.image_extension = ['.png', '.PNG', '.jpg', '.JPG']
        self.transform = transform
        self.img_classification_link = []
        self.pwdf = pd.read_csv(self.classification_datafile_src_path, dtype={'terrainID': str})
        for dirname in os.listdir(self.img_src_folder_path):
            for filename in os.listdir(os.path.join(img_src_folder_path, dirname)):
                if any(filename.endswith(extension) for extension in self.image_extension):
                    terrainID, position, direction = get_info(filename)
                    power_grp = query_power_class(self.pwdf, terrainID, position, direction)
                    if power_grp is None:
                        print("[ERROR] The power group for [terrainID: {}, position: {}, direction: {}] does not "
                                "exist")
                    self.img_classification_link.append(
                        {
                            "img_path": os.path.abspath(os.path.join(
                                self.img_src_folder_path,
                                dirname,
                                filename
                            )), 
                            "power_group": power_grp
                        }
                    )

    def __len__(self):
        """
            This function returns the number of the images in the dataset

            :return: The length of the "images" attribute of the object.
        """
        return len(self.img_classification_link)

    def __getitem__(self, index):
        """
            This function returns a normalized and its corresponding power group

            :param index: The index parameter is the index of the item that needs to be retrieved from the
            dataset. In this case, it is used to retrieve the filepath and power_group of the image at the
            specified index

            :return: A dict containing the normalized heightmap and the power group associated with the
            heightmap at the given index.
        """
        img_path, power_grp = self.img_classification_link[index]['img_path'], \
        self.img_classification_link[index]['power_group']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, power_grp
