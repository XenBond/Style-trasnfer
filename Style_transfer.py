import argparse
import numpy as np
import logging
import sys
import pickle
import cv2

import Transfer
import Blend

def load_image(filename):
    return cv2.imread(filename)

if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Creates a Style-transfer portrait by input an original image and a selected style.')
    parser.add_argument('original_image', type=str, help='The image filename you wanna transfer')
    parser.add_argument('transfered_image', type=str, help='The image filename you wanna save.')
    parser.add_argument('--mask', type=str, nargs='+', help='loading images as masks')
    parser.add_argument('--style', type=int, nargs='+', help='Select styles with the # you wish, Only 1 2 3 available')
    args = parser.parse_args()

    logging.info('Loading original image %s' % (args.original_image))
    org_img = load_image(args.original_image)

    logging.info('Method to create mask')
    mask_type = args.mask

    logging.info('Loading style...')
    sty_list = args.style
    masks = args.mask

    mask_list = []
    for i in masks:
        mask_list.append(load_image(i))

    # make a transferor with all inputs, get transfered img.
    transferor = Transfer.Transferor()
    transferor.get_original(org_img)
    transferor.get_styles(sty_list)
    transferor.transfer_images()
    transfered_imgs = transferor.return_transfered_list()

    # make a blendor with all input, get blended img.
    blendor = Blend.Blendor()
    blendor.get_input(org_img, transfered_imgs)
    blendor.get_mask(mask_list)
    blendor.blend()
    blended_img = blendor.return_output()
    cv2.imwrite(args.transfered_image, blended_img)
