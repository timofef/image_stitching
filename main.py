import cv2
import numpy as np


def run(img1path, img2path):
    img_ = img1path
    img = img2path

    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # cv2.imshow('original img1', cv2.drawKeypoints(img1, kp1, None))
    # cv2.imshow('original img2', cv2.drawKeypoints(img2, kp1, None))
    cv2.imwrite(r'img1_kp.png', cv2.drawKeypoints(img1, kp1, None))
    cv2.imwrite(r'img2_kp.png', cv2.drawKeypoints(img2, kp2, None))

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_paramS = dict(checks=50)
    match = cv2.FlannBasedMatcher(index_params, search_paramS)
    matches = match.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.35*n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       flags=2)

    img3 = cv2.drawMatches(img_, kp1, img, kp2, good,
                           None, **draw_params)
    cv2.imwrite("matched_points.png", img3)
    MIN_MATCH_COUNT = 1

    M = None  # initialize M to None
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        des_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, des_pts, cv2.RANSAC, 5.0)

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    if M is not None:  # check if M is not None before using it
        dst = cv2.warpPerspective(
            img_, M, (img.shape[1]+img_.shape[1], img.shape[0]))
        dst[0:img.shape[0], 0:img.shape[1]] = img
        cv2.imwrite(r'full_img.png', trim(dst))


def trim(Frame):
    if not np.sum(Frame[0]):
        return trim(Frame[1:])
    if not np.sum(Frame[-1]):
        return trim(Frame[:-2])
    if not np.sum(Frame[:, 0]):
        return trim(Frame[:, 1:])
    if not np.sum(Frame[:, -1]):
        return trim(Frame[:, :-2])

    return Frame


def Stitching():
    run(cv2.imread("test_images/mountain/img2.jpg"), cv2.imread("test_images/mountain/img1.jpg"))


if __name__ == "__main__":
    Stitching()
