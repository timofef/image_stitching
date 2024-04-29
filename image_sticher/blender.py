import cv2 as cv
import numpy as np


class Blender:
    def __init__(self, name: str):
        if name == "":
            raise Exception("Blender: name can't be empty")

        if name is None:
            name = "Overlay"

        if name == "Overlay":
            blend = get_overlay_blending_function()
            combine = get_combine_overlay_function()
        elif name == "Alpha":
            blend = get_alpha_blending_function()
            combine = get_combine_alpha_function()
            # combine = get_combine_overlay_function()

        self.name = name
        self.blend = blend
        self.combine = combine

        print("Blender: " + self.name)


# color corrector?


def get_combine_overlay_function():
    def h(base_image, warped_image):
        warped_image[0:base_image.shape[0], 0:base_image.shape[1]] = base_image
        return warped_image

    return h


def get_combine_alpha_function():
    def h(base_image, warped_image):
        blended_image = cv.add(warped_image, np.resize(base_image, warped_image.shape))
        return np.add(warped_image,
                      cv.copyMakeBorder(
                          base_image,
                          0,
                          warped_image.shape[1] - base_image.shape[1],
                          0, warped_image.shape[0] - base_image.shape[0],
                          cv.BORDER_CONSTANT, value=[0, 0, 0]))

        # cv.imshow('Blended Image', cv.copyMakeBorder(base_image, 0, warped_image.shape[0] - base_image.shape[0], 0,
        #                   warped_image.shape[1] - base_image.shape[1], cv.BORDER_CONSTANT, value=[0,0,0]))
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    return h


def get_overlay_blending_function():
    def h(img1, img2):
        return img1, img2

    return h


def get_alpha_blending_function():
    def h(img1, img2):
        alpha = 0.5

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        gray1[gray1 > 0] = 1
        gray2[gray2 > 0] = 1
        gray2_cropp = gray2[0:gray1.shape[0], 0:gray1.shape[1]]

        mask = gray1 & gray2_cropp
        mask = np.logical_not(mask).astype(int)

        n = np.empty((mask.shape[0], mask.shape[1], 3))

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                n[i][j][0] = mask[i][j]
                n[i][j][1] = mask[i][j]
                n[i][j][2] = mask[i][j]
        n[n < 1] = 0.5

        img1 = img1.astype(float)
        img2 = img2.astype(float)

        img1 = cv.multiply(n, img1)
        tmp = cv.multiply(n, img2[0:gray1.shape[0], 0:gray1.shape[1]])
        img2[0:gray1.shape[0], 0:gray1.shape[1]] = tmp

        # cv.imshow('Blended Image', n)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        return img1, img2

    return h
