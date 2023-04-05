import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from lib import *

# get all the folders in the dataset except the ones that start with a dot
folders_list = sorted([folder for folder in os.listdir('resized') if not folder.startswith('.')])
# assign a period to each folder
periods = {1300+idx*25: folder for idx, folder in enumerate(folders_list)}
print(periods)

def process_period(period, folder):
    dataset = get_images(period, folder)
    x_train, y_train, x_test, y_test = train_test_split(dataset[:, 0], dataset[:, 1], test_size=0.2)
    # store x_train under the directory of train
    np.save(f'train/paths/{period}', x_train, allow_pickle=True)
    # store x_train under the directory of test
    np.save(f'test/paths/{period}', x_test, allow_pickle=True)

    image_paths = np.load(f'train/paths/{period}.npy', allow_pickle=True)
    print(f'Loading images for period {period}...')
    binary_images = np.array(list(map(load_image_binarize, image_paths)), dtype=object)
    print(f'Computing features for period {period}...')
    features_of_period = np.array(list(map(get_psd, binary_images)), dtype=object)
    np.save(f'train/features/{period}', features_of_period, allow_pickle=True)

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_period, period, folder) for period, folder in periods.items()]
    for future in futures:
        future.result()