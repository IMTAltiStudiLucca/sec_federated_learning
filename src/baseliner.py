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

def fracture_image(image, num_frac=3):

    perturbation = perturb.Fracture(num_frac)

    morphology = morpho.ImageMorphology(image, scale=4)

    perturbed_hires_image = perturbation(morphology)
    perturbed_image = morphology.downscale(perturbed_hires_image)

    return perturbed_image

#def thinning_image(image, amount=.5):

#    perturbation = perturb.Thinning(amount)

#    morphology = morpho.ImageMorphology(image, scale=4)

#    perturbed_hires_image = perturbation(morphology)
#    perturbed_image = morphology.downscale(perturbed_hires_image)

#    return perturbed_image


#    perturb.Thickening(amount=1.),
#    perturb.Swelling(strength=3, radius=7),

img = io.load_idx("images-idx3-ubyte.gz")[int(sys.argv[1])]

print_digit(img)

print_digit(fracture_image(img, int(sys.argv[2])))
