import cv2 as cv


class Homography:
    def __init__(self, name: str):
        if name == "":
            raise Exception("Homography: name can't be empty")

        if name is None:
            name = "Perspective"

        if name == "Perspective":
            find_homography_function = find_homography_perspective("RANSAC")
            needed_points = 50
            warp_function = cv.warpPerspective
        elif name == "Affine":
            find_homography_function = cv.getAffineTransform
            warp_function = cv.warpAffine
            needed_points = 3
        else:
            raise Exception("Homography: unknown homography name " + name)

        self.name = name
        self.find_homography = find_homography_function
        self.warp = warp_function
        self.needed_points = needed_points

        print("Homography: " + self.name)


def find_homography_perspective(alg:str):
    if alg == "RANSAC":
        def h(src_points, dst_points):
            return cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)[0]

        return h
