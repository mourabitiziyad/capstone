import matplotlib
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from skimage.util import invert
from skimage.morphology import skeletonize
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
from skimage import img_as_ubyte
import concurrent.futures
import os
import cv2
e = 7
min_frag_size = 2*e+2
import time


def skimage_to_opencv(skeleton):
    """Converts a skeleton image from skimage to opencv format."""
    return (skeleton * 255).astype(np.uint8)


def opencv_connected_components(skeleton):
    """Computes connected components using opencv."""
    ret, labels = cv2.connectedComponents(skeleton)
    return labels


def extract_fragments(skeleton, labels):
    """Extracts fragments from a skeleton image and its labels."""
    fragments = [ [] for _ in range(np.max(labels))]
    
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y, x]:
                label = labels[y, x] - 1
                fragments[label].append((y, x))
    return [fragment for fragment in fragments if fragment if len(fragment) > min_frag_size]

from scipy.signal import argrelextrema, savgol_filter, find_peaks
import math
from scipy.ndimage import gaussian_filter, uniform_filter1d
#pixel_to_p2_p3_map = {}  # Mapping data structure to store the pixel to p2 and p3 mapping

def get_adjacent_pixels(skeleton, point):
    h, w = skeleton.shape
    adjacent_pixels = []
    for sy in range(-1, 2):
        for sx in range(-1, 2):
            if 0<= point[0] + sy < h and 0 <= point[1]+sx < w and skeleton[point[0] + sy, point[1] + sx] and not (sy == 0 and sx == 0):
                adjacent_pixels.append([point[0] + sy, point[1] + sx])
    return sorted(adjacent_pixels)


def compute_window_size(array_length, min_array_length, max_array_length, min_window_size=2, max_window_size=50):
    # Linear scaling between min_window_size and max_window_size
    window_size = min_window_size + (max_window_size - min_window_size) * (array_length - min_array_length) / (max_array_length - min_array_length)
    return int(window_size)



def traverse(skeleton, point, visited, e, count=0):
    if e == count:
        return point
    adjacent_pixels = get_adjacent_pixels(skeleton, point)
    if len(adjacent_pixels) != 2:
        return point
    
    visited[tuple(point)] = True
    if not visited[tuple(adjacent_pixels[0])]:
        return traverse(skeleton, adjacent_pixels[0], visited, e, count+1)
    return traverse(skeleton, adjacent_pixels[1], visited, e, count+1)


def find_minima(angles, window_size=10):
    epsilon = 1e-10
    adjusted_angles = angles - np.arange(len(angles)) * epsilon

    # Calculate the number of windows
    num_windows = (len(angles) - window_size + 1 + window_size - 1) // window_size

    # Create an array of window indices
    window_indices = np.arange(num_windows) * window_size

    # Calculate minima indices for each window
    min_angle_indices = window_indices + np.argmin(adjusted_angles[window_indices[:, None] + np.arange(window_size)], axis=1)

    # Filter consecutive indices
    mask = np.concatenate(([True], min_angle_indices[1:] - min_angle_indices[:-1] > 1))
    min_angle_indices_filtered = min_angle_indices[mask]

    return min_angle_indices_filtered

def compute_angle(point, p2, p3):
    angle1 = np.arctan2(point[0] - p2[0], point[1] - p2[1])
    if angle1 < 0.0:
        angle1 += 2 * np.pi
    angle2 = np.arctan2(point[0] - p3[0], point[1] - p3[1])
    if angle2 < 0.0:
        angle2 += 2 * np.pi
    angle = np.abs(angle1 - angle2)
    tang_angle = min(angle, 2 * np.pi - angle)
    return tang_angle

'''
def find_threshold(data, window_size):
    threshold = np.zeros_like(data)
    half_window = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window_data = data[start:end]
        
        median = np.median(window_data)
        deviations = np.abs(window_data - median)
        mad = np.median(deviations)
        threshold_value = median - 2*mad  # Adjust this formula as needed

        threshold[start:end] = threshold_value

    return threshold'''


def plot_high_curvature_points_angles(high_curvature_points_angles):
    plt.figure(figsize=(10, 10))
    plt.plot(range(len(high_curvature_points_angles)), high_curvature_points_angles, linestyle='-')
    plt.xlabel("Index")
    plt.ylabel("Angle (radians)")
    plt.title("High Curvature Points Angles")
    plt.ylim(0, 3 * np.pi)
    plt.show()
    

'''def extract_angles_minima(skeleton, fragments, e):
    smallest_fragment = min(fragments, key=len)
    largest_fragment = max(fragments, key=len)
    minima = []
    for fragment in fragments:
        candidates = []
        for index, point in enumerate(fragment):
            visited = np.zeros_like(skeleton)
            visited[tuple(point)] = True
            
            adjacent_pixels = get_adjacent_pixels(skeleton, point)
            if len(adjacent_pixels) != 2:
                candidates.append(2*np.pi)
                continue
            direction_point, opposite_point = adjacent_pixels
            p2 = traverse(skeleton, direction_point, visited, e)
            p3 = traverse(skeleton, opposite_point, visited, e)
            tang_angle = compute_angle(point, p2, p3)
            
            candidates.append(tang_angle)
        window_size = compute_window_size(len(fragment), len(smallest_fragment), len(largest_fragment))
        fragment_minima = find_minima(candidates, window_size=window_size)

        for index in fragment_minima:
            minima.append(fragment[index])
    return minima'''

def process_point(point, skeleton, e):
    visited = np.zeros_like(skeleton)
    visited[tuple(point)] = True
    adjacent_pixels = get_adjacent_pixels(skeleton, point)

    if len(adjacent_pixels) != 2:
        return 2 * np.pi

    direction_point, opposite_point = adjacent_pixels
    p2 = traverse(skeleton, direction_point, visited, e)
    p3 = traverse(skeleton, opposite_point, visited, e)
    tang_angle = compute_angle(point, p2, p3)
    return tang_angle

def extract_angles_minima(skeleton, fragments, e):
    smallest_fragment = min(fragments, key=len)
    largest_fragment = max(fragments, key=len)
    minima = []

    for fragment in fragments:
        num_points = len(fragment)
        candidates = np.empty(num_points)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            point_results = list(executor.map(process_point, fragment, [skeleton] * num_points, [e] * num_points))

        candidates = np.array(point_results)
        window_size = compute_window_size(num_points, len(smallest_fragment), len(largest_fragment))
        fragment_minima = find_minima(candidates, window_size=window_size)

        minima.extend([fragment[index] for index in fragment_minima])

    return minima

from collections import defaultdict
import numpy as np


def euclidean_distance(p1, p2):
    return np.sqrt((p1[1] - p2[1]) ** 2 + (p1[0] - p2[0]) ** 2)


def bresenham_line(binary_image, x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def stroke_length(binary_image, key_point, sin_theta, cos_theta, max_distance=1000):
    y, x = key_point
    x_e = int(x + max_distance * cos_theta)
    y_e = int(y + max_distance * sin_theta)
    line_points = bresenham_line(binary_image, x, y, x_e, y_e)

    for y_p, x_p in line_points:
        if 0 <= x_p < binary_image.shape[1] and 0 <= y_p < binary_image.shape[0]:
            if binary_image[y_p, x_p]:  # background pixel
                return euclidean_distance((x, y), (x_p, y_p))
        else:
            break

    return 0


def psd_feature_vector(binary_image, key_point, num_directions=72):
    stroke_lengths = np.zeros(num_directions)

    sin_values = np.sin(2 * np.pi * np.arange(num_directions) / num_directions)
    cos_values = np.cos(2 * np.pi * np.arange(num_directions) / num_directions)
    #before = time.time()
    for m in range(num_directions):
        
        stroke_lengths[m] = stroke_length(binary_image, key_point, sin_values[m], cos_values[m])
    #after = time.time()
    #print("Time taken for psd:", after - before)

    epsilon = 1e-9
    feature_vector = stroke_lengths / (np.sum(stroke_lengths) + epsilon)
    max_idx = np.argmax(feature_vector)
    feature_vector = np.roll(feature_vector, -max_idx)  # make it rotational-invariant
    return feature_vector


def get_images(period, folder):

    """returns a list of image paths for a given period"""
    folder_size = len(os.listdir(f'resized/{folder}'))
    # create an array to hold each path along with its label(period)
    image_paths = np.zeros((folder_size, 2), dtype=object)
    index = 0
    for file in os.listdir(f'resized/{folder}'):
        image_paths[index, 0] = f'resized/{folder}/{file}'
        image_paths[index, 1] = period
        index += 1
    return image_paths


def display_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")
    plt.show()


def load_image_binarize(image_path):
    """loads an image from a given path"""
    # Load the image
    image = io.imread(image_path)
    image = color.rgb2gray(image)
    # Apply Sauvola thresholding
    window_size = 25
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    binary_sauvola = image > thresh_sauvola
    return binary_sauvola


def get_forkpoints(skeleton):
    selems = list()
    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
    selems.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
    selems.append(np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]))
    selems = [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]

    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))

    forkpoints = np.zeros_like(skeleton, dtype=bool)
    for selem in selems:
        forkpoints |= ndi.binary_hit_or_miss(skeleton, selem)
    return np.argwhere(forkpoints)



def get_keypoints(skeleton):
    forkpoints = get_forkpoints(skeleton)
    open_cv_skeleton = skimage_to_opencv(skeleton)
    labels = opencv_connected_components(open_cv_skeleton)
    fragments = extract_fragments(skeleton, labels)
    e = 7
    # find the time before the function call
    start = time.time()
    high_curvature_points = extract_angles_minima(skeleton, fragments, e)
    # find the time after the function call
    end = time.time()
    print(f"Time taken by function: {end - start}")
    keypoints = np.concatenate((forkpoints, np.array(high_curvature_points)))
    return keypoints


def compute_feature_vector(args):
    binary_image, keypoint = args
    return psd_feature_vector(binary_image, keypoint)


def get_psd(binary_image):
    print('image')
    skeleton = skeletonize(invert(binary_image))
    keypoints = get_keypoints(skeleton)
    keypoints_count = len(keypoints)
    num_directions = 72  # This should match the value you used in the `psd_feature_vector` function
    feature_vectors = np.zeros((keypoints_count, num_directions))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(compute_feature_vector, [(binary_image, keypoint) for keypoint in keypoints]))

    for idx, result in enumerate(results):
        feature_vectors[idx] = result
    return feature_vectors
