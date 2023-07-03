# Power Predictor Version 2

The power predictor is part of the program I wrote for 'vibration-based harvester simulator' as author's final year project (FYP) for predicting the power generated if the car has the vision of its front view (using Unity3D built-in camera)

An updating version of power predictor is shown in this project. 

## Features of the update
1. The heightmap (W:512 x H:64) regarding each point in an image in Unity unit is collected.
2. The preprocessing module for heightmap (heightmap is recorded in a string format and the module is responsible for transfering the string into appropriate numpy format)
3. The normalization module for heightmap (to normalize each pixel value into the range of [-1, 1]
4. The model for predicting the heightmap from image has been updated to UNet-like structure
