import numpy as np
import cv2

class Blendor:

    def __init__(self):
        self.original_image = None
        self.transfered_img_list = []
        self.mask_list = []
        self.output_img = None

    # get input org & transfered images list
    def get_input(self, org_img, transfered_img_list):
        self.original_image = org_img
        self.transfered_img_list = transfered_img_list

    # TO DO: Open an image as the mask.
    def get_mask(self, mask_list):
        self.mask_list = mask_list

    # TO DO: blend the images with mask.
    def blend(self):
        if self.original_image is None:
            raise ValueError('lack the original image.')
        elif len(self.transfered_img_list)==0:
            raise ValueError('lack the transfered image.')
        elif len(self.mask_list)==0:
            raise ValueError('lack the mask.')
        else:
            # TO DO begins here
            self.output_img = np.copy(self.original_image)
            for i in range(len(self.transfered_img_list)):
                mask = self.mask_list[i]
                mask = mask.astype(np.float) / mask.max()
                foreground = mask * self.transfered_img_list[i]
                background = (1 - mask) * self.output_img
                self.output_img = (foreground + background).astype(np.int)

    # return output image
    def return_output(self):
        if self.output_img is None:
            raise ValueError('Not get blended yet.')
        return self.output_img

    # TO DO: delete existing data from this object
    def conditional_delete(self, threshold=0):
        return 0
