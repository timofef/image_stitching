import cv2
import numpy as np

from image_sticher.feature_detector import FeatureDetector
from image_sticher.feature_matcher import FeatureMatcher
from image_sticher.homography import Homography
from image_sticher.blender import Blender


class PairStitcher:

    def __init__(self,
                 feature_detector: FeatureDetector,
                 feature_matcher: FeatureMatcher,
                 homography: Homography,
                 blender: Blender):
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.homography = homography
        self.blender = blender

        self.save_keypoints = False
        self.save_matches = False

    def stitch(self, center_img, other_img):
        # Load the test_images
        image1 = cv2.imread(other_img)
        image2 = cv2.imread(center_img)

        # Convert test_images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors for both test_images
        keypoints1, descriptors1 = self.feature_detector.detect_and_compute(gray1, None)
        keypoints2, descriptors2 = self.feature_detector.detect_and_compute(gray2, None)
        if self.save_keypoints:
            cv2.imwrite(r'image1_kp.png', cv2.drawKeypoints(image2, keypoints2, None))
            cv2.imwrite(r'image2_kp.png', cv2.drawKeypoints(image1, keypoints1, None))

        # Match the descriptors using brute-force matching
        matches = self.feature_matcher.match(descriptors1, descriptors2)
        # if self.save_matches:
            # draw_params = dict(matchColor=(0, 255, 0),
            #                    singlePointColor=None,
            #                    flags=2)
            # m = cv2.drawMatches(gray2, keypoints2, gray1, keypoints1, matches,None, flags=2)
            # m = cv2.drawMatches(gray2, keypoints2, gray1, keypoints1, matches, None)
            # cv2.imwrite(r'matches.png', m)

        # Select the top N matches
        num_matches = self.homography.needed_points
        matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

        # Extract matching keypoints
        src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

        # Estimate the homography matrix
        homography = self.homography.find_homography(src_points, dst_points)

        # Warp the first image using the homography
        warped_image = self.homography.warp(image1, homography,
                                            (image2.shape[1] + image1.shape[1], image2.shape[0] + image1.shape[0]))

        # Blend images
        image2, warped_image = self.blender.blend(image2, warped_image)
        res = self.blender.combine(image2, warped_image)

        return trim(res)
        # cv2.imwrite(r'conn.jpg', trim(warped_image))
        # cv2.imshow('Blended Image', trim(warped_image))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Blending the warped image with the second image using alpha blending
        # alpha = 0.5  # blending factor
        # blended_image = cv2.addWeighted(warped_second_image, alpha, image2, 1 - alpha, 0)

        # Display the blended image
        # cv2.imshow('Blended Image', blended_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def set_options(self, save_keypoints=False, save_matches=False):
        self.save_keypoints = save_keypoints
        self.save_matches = save_matches
        return self


def trim(frame):
    """Trim black areas on image sides"""
    while not np.sum(frame[0]):
        frame = frame[1:]
    while not np.sum(frame[-1]):
        frame = frame[:-2]
    while not np.sum(frame[:, 0]):
        frame = frame[:, 1:]
    while not np.sum(frame[:, -1]):
        frame = frame[:, :-2]

    return frame
