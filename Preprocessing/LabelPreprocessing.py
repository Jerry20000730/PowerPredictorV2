import os
from typing import Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm


class LabelPreprocessing():
    def __init__(self,
                 data_file_path: str,
                 image_size: Tuple[int, int]):
        self.data_file_path = data_file_path
        self.image_size = image_size

    def _load_file(self, filepath):
        return pd.read_csv(filepath, dtype={'terrainID': str})

    def _save_mat(self, filepath, filename, mat):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        np.save(os.path.join(filepath, filename), mat)

    def _produce_heightmap(self, df):
        for i in range(df.shape[0]):
            if len(df["image_height"][i].strip("[]").split()) != self.image_size[0]*self.image_size[1]:
                print("[Error]: {0}_{1}_{2} does not comply the size of the image"\
                      .format(df['terrainID'][i], df['position'][i], df['direction'][i]))
                continue
            mat = np.array([float(height) - df["current_height"][i] for height in
                            df["image_height"][i].strip("[]").split()]).reshape(self.image_size)
            mat = np.flipud(mat)
            self._save_mat(os.path.join(self.data_file_path, 'Label_heightmap'),
                           '{0}_{1}_{2}_mat.npy'.format(df['terrainID'][i], df['position'][i], df['direction'][i]),
                           mat)

    def start(self):
        files = os.listdir(self.data_file_path)
        with tqdm(total=len(files), position=0) as pbar:
            pbar.set_description("Preprocessing matrix")
            for file in files:
                df = self._load_file(os.path.join(self.data_file_path, file))
                self._produce_heightmap(df)
                pbar.update(1)
        print("[INFO] All heightmap has been stored in {}".format(os.path.join(self.data_file_path, 'Label_heightmap')))
