import sys
import numpy as np
from morphomnist import io, morpho, perturb, util
import matplotlib.pyplot as plt

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

def draw_digit(image,label):
    fig = plt.figure
    plt.imshow(image, cmap='gray')
    name_fig =  "%i.png" %label
    plt.savefig(name_fig, dpi=300)

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

## Cancel the picture from left
def cancelFromLeft(image, alpha):
    assert(0 <= alpha <= 1)
    new_image = squarize(image.copy())
    tbr = len(new_image) * len(new_image[0]) * alpha
    for j in range(len(new_image[0])):
        for i in range(len(new_image)):
            if tbr > 0:
                tbr -= 1
                new_image[i][j] = 0

    return linearize(new_image)


## Cancel the picture from top
def cancelFromTop(image, alpha):
    assert(0 <= alpha <= 1)
    new_image = squarize(image.copy())
    tbr = len(new_image) * len(new_image[0]) * alpha
    for i in range(len(new_image)):
        for j in range(len(new_image[i])):
            if tbr > 0:
                tbr -= 1
                new_image[i][j] = 0

    return linearize(new_image)


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

def squarize(list, width=28, height=28):
    assert(len(list) == width * height)
    matrix = []
    for h in range(height):
        row = []
        for w in range(width):
            row.append(list[(h * width) + w])
        matrix.append(row)
    return matrix


def linearize(matrix):
    list = []
    for i in range(len(matrix)):
        list.extend(matrix[i])
    return list

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

def dotted_zero():
    image = io.load_idx("../data/train-images-idx3-ubyte.gz")[56]
    print(linearize(image))
    image = baseline_set_pixel(image, 255, 16, 16)
    image = baseline_set_pixel(image, 255, 16, 15)
    image = baseline_set_pixel(image, 255, 15, 16)
    image = baseline_set_pixel(image, 255, 15, 15)
    print(linearize(image))
    draw_digit(image,1000)

def slashed_zero():
    image = io.load_idx("../data/train-images-idx3-ubyte.gz")[56]
    label = io.load_idx("../data/train-labels-idx1-ubyte.gz")[56]
    print(linearize(image))
    image = baseline_set_pixel(image, 255, 10, 19)
    image = baseline_set_pixel(image, 255, 11, 18)
    image = baseline_set_pixel(image, 255, 12, 17)
    image = baseline_set_pixel(image, 255, 13, 16)
    image = baseline_set_pixel(image, 255, 14, 15)
    image = baseline_set_pixel(image, 255, 15, 14)
    image = baseline_set_pixel(image, 255, 16, 13)
    image = baseline_set_pixel(image, 255, 17, 12)
    image = baseline_set_pixel(image, 255, 18, 11)
    image = baseline_set_pixel(image, 255, 19, 10)
    image = baseline_set_pixel(image, 255, 11, 19)
    image = baseline_set_pixel(image, 255, 12, 18)
    image = baseline_set_pixel(image, 255, 13, 17)
    image = baseline_set_pixel(image, 255, 14, 16)
    image = baseline_set_pixel(image, 255, 15, 15)
    image = baseline_set_pixel(image, 255, 16, 14)
    image = baseline_set_pixel(image, 255, 17, 13)
    image = baseline_set_pixel(image, 255, 18, 12)
    image = baseline_set_pixel(image, 255, 19, 11)
    image = baseline_set_pixel(image, 255, 20, 10)
    image = baseline_set_pixel(image, 255, 12, 19)
    image = baseline_set_pixel(image, 255, 13, 18)
    image = baseline_set_pixel(image, 255, 14, 17)
    image = baseline_set_pixel(image, 255, 15, 16)
    image = baseline_set_pixel(image, 255, 16, 15)
    image = baseline_set_pixel(image, 255, 17, 14)
    image = baseline_set_pixel(image, 255, 18, 13)
    image = baseline_set_pixel(image, 255, 19, 12)
    image = baseline_set_pixel(image, 255, 20, 11)
    image = baseline_set_pixel(image, 255, 21, 10)

    print(linearize(image))
    print_digit(image)
    draw_digit(image,label)

def extract_images(n, m, l):
    for i in range(n, m):
        image = io.load_idx("../data/train-images-idx3-ubyte.gz")[i]
        label = io.load_idx("../data/train-labels-idx1-ubyte.gz")[i]
        if label == l:
            draw_digit(image, i)

if __name__ == "__main__":
    # extract_images(0, 1000, 8)
    image = io.load_idx("../data/train-images-idx3-ubyte.gz")[404]
    # draw_digit(cancelFromLeft(image,0.6), 123)
    original = squarize(cancelFromLeft(linearize(image),0.306796875))
    print_digit(original)
    draw_digit(image, 1234)
    draw_digit(original, 1235)
    # print(linearize(image))
    # dotted_zero()
