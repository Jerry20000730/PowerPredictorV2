# Power Predictor Version 2

The power predictor is part of the program I wrote for 'vibration-based harvester simulator' as author's final year project (FYP) for predicting the power generated if the car has the vision of its front view (using Unity3D built-in camera)

An updating version of power predictor is shown in this project. 

## Features of the update
1. The heightmap (W:512 x H:64) regarding each point in an image in Unity unit is collected.
2. The preprocessing module for heightmap (heightmap is recorded in a string format and the module is responsible for transfering the string into appropriate numpy format)
3. The normalization module for heightmap (to normalize each pixel value into the range of [-1, 1]
4. The model for predicting the heightmap from image has been updated to UNet-like structure

## Function of each folder
1. `Dataset/ImageCollection`: the image collected in Unity
2. `Dataset/PowerCollection`: average power collected and the heightmap stored in csv
3. `Model/`: store the structure of the predictor model
4. `Preprecessing/`: store the utility for preprocessing, including:
    1. extracting matrix data from csv file
    2. normalize matrix data
    3. power classification

## How to run
In `main.py`, uncomment each block of code to implement different task. In github, the power classification task has been finished, meaning that the `STEP3` is not necessary.
```python
from Preprocessing import *
from Model import *

if __name__ == '__main__':
    """
    STEP1: preprocessing the matrix recorded in the csv file
    store the matrix in a specific location 
    """
    # LP = LabelPreprocessing.LabelPreprocessing(
    #     data_file_path='Dataset/PowerCollection',
    #     image_size=(512, 64)
    # )
    # LP.start()
    """
    STEP2: normalize the value
    """
    # norm = Normalization.HeightNormalization(
    #     data_file_path='Dataset/PowerCollection',
    #     custom_path=False
    # )
    # norm.start()
    """
    STEP3: classify the power group
    redistribute the images to specific location
    """
    # pp = PowergroupPreprocessing.PowergroupPreprocessing(
    #     power_data_csv_path='Dataset/PowerCollection',
    #     src_folder_path='Dataset/ImageCollection',
    #     train_dst_folder_path='Train',
    #     test_dst_folder_path='Test',
    #     train_size=10000,
    #     num_classes=5,
    #     after_processing_data_file='Dataset/data_after_preprocessed.csv',
    #     multi_csv=True
    # )
    # pp.start()
    """
    STEP4: train the model using 
    the image collected and
    its correponding heightmap
    """
    predictor = Predictor.EnergyPredictorV2_image2heightmap(
        train_src_folder_path='Train',
        test_src_folder_path='Test',
        heightmap_src_folder_path='Dataset/PowerCollection/Label_heightmap_normalized',
        checkpoint_folder_path='Checkpoint'
    )
    predictor.start()
```
