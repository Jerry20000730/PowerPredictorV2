from Preprocessing import *
from Model import *
from Log import *

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
    # predictor1 = Predictor.EnergyPredictorV2_image2heightmap(
    #     train_src_folder_path='Train',
    #     test_src_folder_path='Test',
    #     heightmap_src_folder_path='Dataset/PowerCollection/Label_heightmap_normalized',
    #     checkpoint_folder_path='Checkpoint',
    #     train_batch_size=4,
    #     test_batch_size=4,
    #     num_epochs=200,
    #     device = 'cuda:1'
    # )
    # predictor1.start()
    """
    STEP5: train the model using
    the heightmap collected and
    its correponding power classification
    """
    # predictor2 = Predictor.EnergyPredictorV2_heightmap2powerclass(
    #     train_src_folder_path='Train',
    #     test_src_folder_path='Test',
    #     heightmap_src_folder_path='Dataset/PowerCollection/Label_heightmap_normalized',
    #     classification_datafile_src_path='Dataset/data_after_preprocessed.csv',
    #     checkpoint_folder_path='Checkpoint',
    #     train_batch_size=16,
    #     test_batch_size=16,
    #     num_epochs=500,
    #     num_classes=5,
    # )
    # predictor2.start()
    """
    ALTERNATIVE STEP: train the model directly
    using the image collected and its
    corresponding power classification
    """
    predictor = Predictor.EnergyPredictorV2_image2powerclassification(
        train_src_folder_path='Train',
        test_src_folder_path='Test',
        classification_datafile_src_path='Dataset/data_after_preprocessed.csv',
        checkpoint_folder_path='Checkpoint',
        train_batch_size=16,
        test_batch_size=16,
        num_epochs=500,
        num_classes=5
    )
    predictor.start()