import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import json
import warnings
warnings.filterwarnings('ignore')
import gc
import sys

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda')


if __name__ == "__main__":

    data_path = "data/Training_Data_RGC_3_55_500K.parquet"
    version = 'RGC_MLP_3_55_V1'
    model_dir = f"Model_Performance/{version}"
    performance_dir = f"Model_Performance/{version}"