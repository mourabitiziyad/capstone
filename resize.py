from PIL import Image
import os

def resize_image(image):
    width, height = image.size
    new_width = int(width//2)
    new_height = int(height//2)
    return image.resize((new_width, new_height), Image.ANTIALIAS)

def main():
    periods = [1300 + i*25 for i in range(0, 11)]
    for period in periods:
        folder = f'Download/MPS{period}-ppm'
        print(f'Processing {folder}')
        for filename in os.listdir(folder):
            if filename.endswith('.ppm'):
                image = Image.open(f'{folder}/{filename}')
                image = resize_image(image)
                save_folder = f'resized/{period}'
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                image.save(f'{save_folder}/{filename}')

if __name__ == '__main__':
    main()