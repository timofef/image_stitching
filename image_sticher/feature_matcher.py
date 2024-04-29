import cv2 as cv


class FeatureMatcher:
    def __init__(self, name: str, binary=False):
        if name == "":
            raise Exception("FeatureMatcher: name can't be empty")

        if name is None:
            name = "BF"

        if name == "BF":
            # match = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE).match
            match = get_bf_match_function(binary)
        elif name == "Flann":
            match = get_flann_match_function(binary)
        else:
            raise Exception("FeatureMatcher: unknown feature matcher name " + name)

        self.name = name
        self.match = match

        print("FeatureMatcher: " + self.name)


def get_bf_match_function(binary):
    def h(descriptors1, descriptors2):
        if binary:
            matcher = cv.BFMatcher(cv.NORM_HAMMING)
        else:
            matcher = cv.BFMatcher(cv.NORM_L2)
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
        ratio_thresh = 0.99
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        return good_matches

    return h


def get_flann_match_function(binary):
    def h(descriptors1, descriptors2):
        if binary:
            index_params = dict(algorithm=6,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=2)
            search_params = {}
        else:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
        matcher = cv.FlannBasedMatcher(index_params, search_params)

        knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
        ratio_thresh = 0.99
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        return good_matches

    return h
