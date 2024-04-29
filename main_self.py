import time

import cv2 as cv

import image_sticher.pair_stitcher as st


def main():
    stitcher = st.PairStitcher(
        st.FeatureDetector("SIFT"),
        st.FeatureMatcher("Flann", binary=False),
        st.Homography("Perspective"),
        st.Blender("Alpha")
    )

    start_time = time.time()
    res = stitcher.\
        set_options(save_keypoints=True, save_matches=True).\
        stitch("test_images/img2.jpg", "test_images/img3.jpg")
    print("--- %s seconds ---" % (time.time() - start_time))

    cv.imwrite(r'conn.jpg', res)


if __name__ == "__main__":
    main()
