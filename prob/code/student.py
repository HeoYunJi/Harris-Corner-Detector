import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy import signal, ndimage
from skimage.color import rgb2gray
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops


def gradient_x(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return signal.convolve2d(image, kernel_x, mode='same')


def gradient_y(image):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return signal.convolve2d(image, kernel_y, mode='same')


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    Harris corner detector를 구현하세요. 수업시간에 배운 간단한 형태의 알고리즘만 구현해도 됩니다.
    스케일scale, 방향성orientation 등은 추후에 고민해도 됩니다.
    Implement the Harris corner detector (See Szeliski 4.1.1).
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    원한다면 다른 종류의 특징점 정합 기법을 구현해도 됩니다.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    만약 영상의 에지 근처에서 잘못된 듯한 특징점이 도출된다면 에지 근처의 특징점을 억제해 버리는 코드를 추가해도 됩니다.
    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    유용한 함수: 제시해 드리는 모든 함수를 꼭 활용해야 하는 건 아닙니다만, 이중 일부가 여러분에게 유용할 수 있습니다.
    각 함수는 웹에서 제공되는 documentation을 참고해서 사용해 보세요.
    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. 

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :입력 인자params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :반환값returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :옵션으로 추가할 수 있는 반환값optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    image = ndimage.gaussian_filter(image, sigma=2)

    Ix = gradient_x(image)
    Iy = gradient_y(image)  # get image's gradient

    IxIx = Ix**2
    IxIy = Ix*Iy
    IyIy = Iy**2
    """    
    IxIx = ndimage.gaussian_filter(Ix ** 2, sigma=1)
    IxIy = ndimage.gaussian_filter(Iy * Ix, sigma=1)
    IyIy = ndimage.gaussian_filter(Iy ** 2, sigma=1)  # gaussian filter with width 1
    """
    height, width = image.shape[:2]

    offset = int(feature_width / 2)

    r = np.zeros(image.shape)

    for y in tqdm(range(offset, height - offset), desc = "computing M components"):
        for x in range(offset, width - offset):
            window_IxIx = IxIx[y - offset:y + offset + 1, x - offset:x + offset + 1]
            window_IyIy = IyIy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            window_IxIy = IxIy[y - offset:y + offset + 1, x - offset:x + offset + 1]

            Mxx = window_IxIx.sum()
            Myy = window_IyIy.sum()
            Mxy = window_IxIy.sum()  # compute M components as squares of deirivatives

            det = Mxx * Myy - Mxy ** 2
            trace = Mxx + Myy

            r[y, x] = det - 0.04 * (trace ** 2)  # compute cornerness

    cv.normalize(r, r, 0.0, 1.0, cv.NORM_MINMAX)

    xs = np.zeros(1)
    ys = np.zeros(1)

    for y in tqdm(range(offset, height - offset), desc = "detecting corners"):
        for x in range(offset, width - offset):
            if r[y, x] > 0.2:   # Threshold on 0.2 to pick high cornerness
                xs = np.append(xs, x)
                ys = np.append(ys, y)

    # Non-maximal suppression to pick peaks ?
    """
    scalex = (xs.max() - xs.min()) * 0.5 + xs.min()
    scaley = (ys.max() - ys.min()) * 0.5 + ys.min()

    xs[xs < scalex] = 0
    ys[ys < scaley] = 0
    """
    return xs, ys


"""
##### FOR NEXT ASSIGNMENT ... #####
def get_features(image, x, y, feature_width):
    '''
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # This is a placeholder - replace this with your features!
    features = np.zeros((1,128))

    return features

##### FOR NEXT ASSIGNMENT ... #####
def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!

    matches = np.zeros((1,2))
    confidences = np.zeros(1)


    return matches, confidences
"""
