#Importing all the libraries
from tensorflow.keras.datasets.mnist import load_data
import cv2 
import numpy as np
import os 
import argparse
import math, random
from typing import List, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed

#Creatingg folders to store data
os.makedirs(os.path.join('dataset', 'train'), exist_ok=True)
os.makedirs(os.path.join('dataset', 'test'), exist_ok=True)

#This function is used to fix the pixel values of the image
#It converts all the pixel values to either 0 or 255 i.e perfect black or white
def pixel_fixer_thresh(img, threshold):
    new_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    all_black_pixels = np.where(img < threshold)
    all_white_pixels = np.where(img >= threshold)
    new_image[all_black_pixels] = 0
    new_image[all_white_pixels] = 255
    return new_image

#Generate random lines on the input images
def noise_image(blank_image,img_size):
    for _ in range(random.randint(5,12)):
        pt1 = (random.randint(0, img_size), random.randint(0, img_size))
        pt2 = (random.randint(0, img_size), random.randint(0, img_size))
        cv2.line(blank_image, pt1, pt2, (255,255,255), 1)
    blank_image = pixel_fixer_thresh(blank_image,2)
    return blank_image
    
#Resize images without loosing the aspect ratio
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

#This function is used to generate the dataset
def generate_image(image, imsize,fname, dtype='train'):
    digit = image.astype(np.float32)
    digit = image_resize(digit, width=imsize, height=imsize)
    digit = digit.astype(np.uint8)
    digit = pixel_fixer_thresh(digit,2)
    output_image  = digit.copy()
    digit = cv2.Canny(image=digit, threshold1=1, threshold2=1, apertureSize=5, L2gradient=True)
    input_image = noise_image(digit,imsize)
    input_image = pixel_fixer_thresh(input_image,2)
    savepath = os.path.join('dataset', dtype, f'{fname}.png')
    #merge both the input and output image 
    merged_image = np.concatenate((input_image, output_image), axis=1)
    cv2.imwrite(savepath, merged_image)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size', type=int, default=2500)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=-1)
    args = parser.parse_args()
    #Loading the MNIST dataset from TF2 library
    (x_train, y_train), (x_test, y_test) = load_data()
    #Multiprocessing the dataset generation of training and testing dataset
    _ = Parallel(n_jobs=args.num_workers)(delayed(generate_image)(x_train[i], args.image_size, str(i), 'train') for i in tqdm(range(args.dataset_size)))
    test_size = int(args.dataset_size * 0.1)
    _ = Parallel(n_jobs=args.num_workers)(delayed(generate_image)(x_test[i], args.image_size, str(i), 'test') for i in tqdm(range(test_size)))
    print('Data generated!')