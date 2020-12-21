import sys
import numpy as np
from morphomnist import io, morpho, perturb, util

def byte_to_char(byte):
    if byte < 50:
        return ' '
    elif byte < 100:
        return '.'
    elif byte < 150:
        return 'x'
    elif byte < 200:
        return 'X'
    else:
        return '#'

def print_top(width=28):
    top = "+"
    for w in range(width):
        top = top + "-"
    top = top + "+"
    print(top)

def print_digit(image_matrix, width=28, height=28):
    print_top()
    for h in range(height):
        row = "|"
        for w in range(width):
            row = row + byte_to_char(image_matrix[h][w])
        row = row + "|"
        print(row)
    print_top()

## Use Morphmnist to fracture a picture
def baseline_fracture(image, num_frac=3):

    perturbation = perturb.Fracture(num_frac)

    morphology = morpho.ImageMorphology(image, scale=4)

    perturbed_hires_image = perturbation(morphology)
    perturbed_image = morphology.downscale(perturbed_hires_image)

    return perturbed_image


## Set a specific (x,y) pixel to byte
def baseline_set_pixel(image, byte, x, y):
    new_image = image.copy()
    new_image[x][y] = byte
    return new_image

## sums a constant byte value to image
def baseline_shift_color(image, byte):
    new_image = image.copy()
    for i in range(len(image)):
        for j in range(len(image[i])):
            new_image[i][j] += byte

    return new_image

## sums a mask of bytes to image (alpha channel is for transparency)
def baseline_overlay_mask(image, mask, alpha=1):
    new_image = image.copy()
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            new_image[i][j] += int(round(mask[i][j] * alpha))

    return new_image


img = io.load_idx("../data/train-images-idx3-ubyte.gz")[int(sys.argv[1])]

print_digit(img)

print_digit(baseline_shift_color(img, 70))
