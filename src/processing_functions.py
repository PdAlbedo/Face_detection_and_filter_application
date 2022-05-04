"""
CS5330 - Pattern Recognition & Computer Vision
Project Title: Face detection and filter application
April 18, 2022
Team: Sida Zhang, Xichen Liu, Xiang Wang, Hongyu Wan

Description: Basic calculations and formulas used in the program.
Including but not limit to: face drawing, transofrmation,
ssd and knn where the k parameter is 1, and building embedding space
"""
__author__ = "Sida Zhang, Hongyu Wan, Xiang Wang, Xichen Liu"

import cv2
import torch
import numpy as np
import gui_functions

torch.manual_seed(888)

COLOUR_CORRECT_BLUR_FRAC = 0.6
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11


# draw face area
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color = color)


# correct colours for two detected faces
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[gui_functions.LEFT_EYE_POINTS], axis = 0) -
        np.mean(landmarks1[gui_functions.RIGHT_EYE_POINTS], axis = 0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


# warp two faces
def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype = im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst = output_im,
                   borderMode = cv2.BORDER_TRANSPARENT,
                   flags = cv2.WARP_INVERSE_MAP)
    return output_im


# get face mask area
def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype = np.float64)

    for group in gui_functions.OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color = 1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation.

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis = 0)
    c2 = np.mean(points2, axis = 0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


# calculate ssd
def ssd(a, b):
    d = np.sum((a - b) ** 2)
    return d


# matching to minimize ssd distance image
def nn(results, targets, a):
    a = a.cpu().detach().numpy()
    targets = np.array(targets)
    min_dis = float('inf')
    file_name = []
    for i in range(len(results)):
        d = ssd(a, results[i])
        if d < min_dis:
            min_dis = d
            file_name = targets[i]
    return file_name


# built embedding space
def build_embedding_space(model, dataloader):
    model.eval()
    results = []
    targets = []
    b = 0
    for data, target in dataloader:
        output = model(data)
        print("\nBatch %d:" % b)
        print("Input batch size: ", end = "")
        print(data.shape)
        print("Apply the model with 50-node dense layer to the data, "
              "we have the returned output with the shape of: ", end = "")
        print(output.shape)
        b += 1

        for i in range(len(output)):
            results.append(output[i])
            targets.append(target[i])
    print("\nShape of the output nodes from the model: ", end = "")
    print(torch.stack(results).shape)

    return results, targets
