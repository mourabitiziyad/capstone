from PIL import Image
import os

def resize_image(image, new_width):
    width, height = image.size
    aspect_ratio = float(width) / float(height)
    new_height = int(new_width / aspect_ratio)
    return image.resize((new_width, new_height), Image.ANTIALIAS)


def main():
    periods = [1300 + i*25 for i in range(0, 1)]
    for period in periods:
        folder = f'dev_images/{period}'
        print(f'Processing {folder}')
        for filename in os.listdir(folder):
            if filename.endswith('.ppm'):
                image = Image.open(f'{folder}/{filename}')
                width, _ = image.size
                image = resize_image(image, width//3)
                save_folder = f'resized/{period}'
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                image.save(f'{save_folder}/{filename}')

if __name__ == '__main__':
    main()