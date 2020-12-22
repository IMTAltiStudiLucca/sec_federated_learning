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

## modifies image brightness accoridng to byte
def baseline_change_brightness(image, byte):
    new_image = image.copy()
    for i in range(len(image)):
        for j in range(len(image[i])):
            val = new_image[i][j] + byte
            if val > 255:
                val = 255
            elif val < 0:
                val = 0
            new_image[i][j] = val

    return new_image

## shift position
def baseline_shift_position(image, hshift, vshift):
    new_image = image.copy()
    H = len(image)
    for i in range(H):
        W = len(image[i])
        for j in range(W):
            new_image[(i + vshift) % H][(j + hshift) % W] = image[i][j]

    return new_image

## sums a mask of bytes to image (alpha channel is for transparency)
def baseline_overlay_mask(image, mask, alpha=1):
    new_image = image.copy()
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            new_image[i][j] += int(round(mask[i][j] * alpha))

    return new_image

def showcase(image):
    print("ORIGINAL")
    print_digit(image)

    print("FRACTURE 3")
    print_digit(baseline_fracture(image))

    print("BLACK CORNER (27,27)")
    print_digit(baseline_set_pixel(image, 255, 27, 27))

    print("COLOR SHIFT +127")
    print_digit(baseline_shift_color(image, 127))

    print("BRIGHTNESS REDUCED BY 70")
    print_digit(baseline_change_brightness(image, -70))

    print("SPATIAL SHIFT BY 3, 7")
    shifted = baseline_shift_position(image, 3, 7)
    print_digit(shifted)

    print("OTHER PICTURE OVERLAY (alpha = 0.2)")
    print_digit(baseline_overlay_mask(image,shifted, .2))


if __name__ == "__main__":
    image = io.load_idx("../data/train-images-idx3-ubyte.gz")[int(sys.argv[1])]
    showcase(image)
