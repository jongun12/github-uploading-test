import os
from importlib import reload
from PIL import Image

import cambridge
reload(cambridge)

dataset_root = '/Users/82105/Desktop/ShopFacade'
dataset = cambridge.CambridgeDataset(dataset_root, mode='train')

image, tr, rot = dataset[0]