import cv2 as cv


class FeatureDetector:
    def __init__(self, name: str):
        if name == "":
            raise Exception("FeatureDetector: name can't be empty")

        if name is None:
            name = "SIFT"

        if name == "SIFT":
            detect_and_compute = cv.SIFT_create().detectAndCompute
        elif name == "ORB":
            detect_and_compute = cv.ORB_create().detectAndCompute
        elif name == "BRISK":
            detect_and_compute = cv.BRISK_create().detectAndCompute
        elif name == "FAST":
            detect_and_compute = get_fast_with_descriptor_detector("SIFT")
        else:
            raise Exception("FeatureDetector: unknown detector name: " + name)

        self.name = name
        self.detect_and_compute = detect_and_compute

        if name == "ORB" or name == "BRISK":
            self.is_binary = True

        print("FeatureDetector: " + self.name)


def get_fast_with_descriptor_detector(descriptor_name):
    detector = cv.FastFeatureDetector_create()

    if descriptor_name == "SIFT":
        descriptor = cv.SIFT_create()
    elif descriptor_name == "ORB":
        descriptor = cv.ORB_create()
    elif descriptor_name == "BRISK":
        descriptor = cv.BRISK_create()
    else:
        raise Exception("FeatureDetector: unknown descriptor name for FAST: " + descriptor_name)

    def h(img, mask):
        keypoints = detector.detect(img)
        return descriptor.compute(img, keypoints)

    return h
