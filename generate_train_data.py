from lib import *
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# get all the folders in the dataset except the ones that start with a dot
folders_list = sorted([folder for folder in os.listdir('dev_images') if not folder.startswith('.')])
# assign a period to each folder
periods = {1300+idx*25: folder for idx, folder in enumerate(folders_list[:1])}
print(periods)

def process_period(period, folder):
    dataset = get_images(period, folder)
    x_train, y_train, x_test, y_test = train_test_split(dataset[:, 0], dataset[:, 1], test_size=0.2)
    # store x_train under the directory of train
    np.save(f'train/paths/{period}', x_train, allow_pickle=True)
    # store x_train under the directory of test
    np.save(f'test/paths/{period}', x_test, allow_pickle=True)

with ThreadPoolExecutor() as executor:
    executor.map(process_period, periods.keys(), periods.values())

periods = [1300 + index * 25 for index in range(1)]

def process_period_images(period):
    starttime = time.time()
    image_paths = np.load(f'train/paths/{period}.npy', allow_pickle=True)
    print(image_paths)
    print(f'Loaded {len(image_paths)} images for period {period}')
    print(f'Loading images for period {period}...')
    binary_images = np.array(list(map(load_image_binarize, image_paths)), dtype=object)
    print(f'Computing features for period {period}...')
    features_of_period = np.array(list(map(get_psd, binary_images)), dtype=object)
    np.save(f'train/features/{period}', features_of_period, allow_pickle=True)
    print(f'Finished processing period {period} in {time.time() - starttime} seconds')


process_period_images(periods[0])
# with ThreadPoolExecutor() as executor:
#     executor.map(process_period_images, periods)