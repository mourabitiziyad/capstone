import os
import cv2

# set the path to the directory containing the images
path = "dev_images/1300"

# set the desired maximum size of the images after resizing
max_size = 512

# loop through all the files in the directory
for filename in os.listdir(path):
    # check if the file is an image file
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.ppm'):
        # read the image file
        img = cv2.imread(os.path.join(path, filename))
        # get the dimensions of the original image
        height, width, _ = img.shape
        # calculate the aspect ratio of the original image
        aspect_ratio = width / height
        # calculate the new dimensions based on the aspect ratio and the desired maximum size
        if width > height:
            new_width = max_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(new_height * aspect_ratio)
        # resize the image
        img_resized = cv2.resize(img, (new_width, new_height))
        # write the resized image to a new file
        cv2.imwrite(os.path.join(path, 'resized_' + filename), img_resized)