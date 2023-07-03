import os
from tqdm import tqdm
import numpy as np

class HeightNormalization():
    def __init__(self, data_file_path, custom_path=False):
        self.data_file_path = data_file_path
        # if the custom path is false
        # meaning that the program will
        # look for 'Label_heightmap' under
        # the data_file_folder
        # if the custom path is true
        # the data_file_path will be used as absolute
        # folder for searching
        self.custom_path = custom_path
        self.max_height = -np.Inf
        self.min_height = np.Inf

    def _load_matrix(self, filepath):
        return np.load(filepath)

    def _save_mat(self, filepath, filename, mat):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        np.save(os.path.join(filepath, filename), mat)

    def _find_max(self, np_mat):
        return np_mat.max()

    def _find_min(self, np_mat):
        return np_mat.min()

    def _norm_min_max(self, nd_array, min_val, max_val):
        nd_array = -1 + (2 * (nd_array - min_val) / (max_val - min_val))
        return nd_array

    def norm_min_max(self):
        if self.custom_path:
            filepath = self.data_file_path
        else:
            filepath = os.path.join(self.data_file_path, 'Label_heightmap')

        filenames = os.listdir(filepath)

        with tqdm(total=len(filenames), position=0) as pbar:
            pbar.set_description("Normalization: finding max/min elements")
            for file in filenames:
                mat = np.load(os.path.join(filepath, file))
                max_element = self._find_max(mat)
                min_element = self._find_min(mat)
                if max_element > self.max_height:
                    self.max_height = max_element
                    # print("new max: {}, the max value is: {}".format(file, max_element))
                if min_element < self.min_height:
                    self.min_height = min_element
                    # print("new min: {}, the min value is: {}".format(file, min_element))
                pbar.update(1)
            print("[INFO] the largest height element is {:.2f} and the smallest height element is {:.2f}".format(self.max_height, self.min_height))

        print("[INFO] start normalizing")
        with tqdm(total=len(filenames), position=0) as pbar:
            pbar.set_description("Normalization: using max/min strategy")
            for file in filenames:
                mat = np.load(os.path.join(filepath, file))
                mat = self._norm_min_max(mat, min_val=-30, max_val=30)
                self._save_mat(os.path.join(self.data_file_path, 'Label_heightmap_normalized'),
                               file.split(".")[0] + "_normalized.npy",
                               mat)
                pbar.update(1)
            print("[INFO] All normalized heightmap has been stored in {}".format(
                os.path.join(self.data_file_path, 'Label_heightmap_normalized')))

    def start(self):
        self.norm_min_max()





