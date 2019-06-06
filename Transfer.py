import transfer_func

class Transferor:

    def __init__(self):
        self.original_image = None
        self.styles = []
        self.transfered_img_list = []
        self.CNN_path = ['', 'CNN/', 'CNN1/', 'CNN2/']

    # input original image as np.array
    def get_original(self, org_img):
        del self.original_image
        self.original_image = org_img

    # get style image as np.array
    def get_styles(self, styles_list):
        del self.styles[:]
        for style in styles_list:
            self.styles.append(style)

    # TO DO: transfer the image.
    def transfer_images(self):
        if self.original_image is None:
            raise ValueError('lack the original image.')
        elif len(self.styles) == 0:
            raise ValueError('lack the style image.')
        else:
            del self.transfered_img_list[:]
            for style in self.styles:
                print('Style: ', style)
                self.transfered_img_list.append(transfer_func.transfer(self.original_image, self.CNN_path[style]))

    def return_transfered_list(self):
        if self.transfered_img_list is None:
            raise ValueError('Not get transfered yet.')
        return self.transfered_img_list

    # TO DO: delete existing data from this object
    def conditional_delete(self, threshold=0):
        return 0

