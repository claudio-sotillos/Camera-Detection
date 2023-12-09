import pandas as pd
import os

from .version import Version
from .base import BaseDataset
from .info import DatasetInfo
from . import constants as C


def get_dataset(split="test"): # no need to change this, cause it is changed within lightning/detection testloader
    # meta: path for dataset



    train_path = "/home/hinux/Desktop/project04/checking_folder/PROTEIN_REAL_511_80.csv"


    # tester_path = "/home/hinux/Desktop/project04/ssc/data/Vedran_images/VEDRAN_IMAGES_OG.csv"
    tester_path = "/home/hinux/Desktop/project04/checking_folder/TEST_200.csv"
    # tester_path = "/home/hinux/Desktop/google_images/found_images_9.csv"

    meta = pd.read_csv(tester_path)  # "../data/meta1.csv" #training # "../data/my.csv"#testing   # this changed to the csv containing the path to the images
    info = DatasetInfo.load("../data/info.yaml")
    return BaseDataset(info, meta)[split]
