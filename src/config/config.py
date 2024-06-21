import os
import pathlib
import src
from keras.optimizers import RMSprop

epochs = 500

mb_size = 2

optimizer = RMSprop()

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"dataset")

SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")
