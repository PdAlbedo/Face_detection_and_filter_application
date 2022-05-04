"""
CS5330 - Pattern Recognition & Computer Vision
Project Title: Face detection and filter application
April 18, 2022
Team: Sida Zhang, Xichen Liu, Xiang Wang, Hongyu Wan

Description: Functionalities of the program including detecting face,
greyscale, snapshot, and filters application
"""
__author__ = "Sida Zhang, Hongyu Wan, Xiang Wang, Xichen Liu"

import cv2
import dlib
import math
import torch
import imutils
import numpy as np
import processing_functions
import gui_class

torch.manual_seed(888)

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)


# Grayscale function
def get_gray(input_):
    print('start grayscale')
    output = cv2.cvtColor(input_, cv2.COLOR_BGR2GRAY)
    return output


# Facedetect function
def get_facedetect(input_):
    print('start face detect')
    output = input_
    gray = cv2.cvtColor(src = input_, code = cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = DETECTOR(gray)

    for face in faces:
        # Create landmark object
        landmarks = PREDICTOR(image = gray, box = face)

        # Loop through all the points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # Draw a circle
            cv2.circle(img = output, center = (x, y), radius = 1, color = (0, 255, 0), thickness = -1)
    return output, faces

# Face detect function without draw circle
def get_facedetect_nodraw(input_):
    print('start face detect')
    output = input_
    gray = cv2.cvtColor(src = input_, code = cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = DETECTOR(gray)

    for face in faces:
        # Create landmark object
        landmarks = PREDICTOR(image = gray, box = face)

        # Loop through all the points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
    return output, faces

# Face Swap function
def get_exchange_face(input_, faces):
    output = input_
    if len(faces) < 2:
        print("not enough faces")

    else:
        landmarks1 = np.matrix([[p.x, p.y] for p in PREDICTOR(input_, faces[0]).parts()])
        landmarks2 = np.matrix([[p.x, p.y] for p in PREDICTOR(input_, faces[1]).parts()])

        M1 = processing_functions.transformation_from_points(landmarks1[ALIGN_POINTS],
                                                             landmarks2[ALIGN_POINTS])

        M2 = processing_functions.transformation_from_points(landmarks2[ALIGN_POINTS],
                                                             landmarks1[ALIGN_POINTS])

        mask1 = processing_functions.get_face_mask(input_, landmarks2)
        mask2 = processing_functions.get_face_mask(input_, landmarks1)
        warped_mask1 = processing_functions.warp_im(mask1, M1, input_.shape)
        warped_mask2 = processing_functions.warp_im(mask2, M2, input_.shape)
        combined_mask1 = np.max([processing_functions.get_face_mask(input_, landmarks1), warped_mask1],
                                axis = 0)
        combined_mask2 = np.max([processing_functions.get_face_mask(input_, landmarks2), warped_mask2],
                                axis = 0)
        warped_im1 = processing_functions.warp_im(input_, M1, input_.shape)
        warped_im2 = processing_functions.warp_im(input_, M2, input_.shape)
        warped_corrected_im1 = processing_functions.correct_colours(input_, warped_im1, landmarks1)
        warped_corrected_im2 = processing_functions.correct_colours(input_, warped_im2, landmarks2)

        output_im = input_ * (1.0 - combined_mask1) + warped_corrected_im1 * combined_mask1
        output_im = output_im * (1.0 - combined_mask2) + warped_corrected_im2 * combined_mask2

        cv2.imwrite('output.jpg', output_im)

        output = output_im.astype(np.uint8)
    return output


# Face Modifications function
def get_filtered(input_, faces):
    print('start filter')
    output = input_
    gray = cv2.cvtColor(src = input_, code = cv2.COLOR_BGR2GRAY)
    if gui_class.if_glass:
        if len(faces) == 0:
            print("There is no face detected!")
        else:
            glasses = cv2.imread("sun_glasses.png", cv2.IMREAD_UNCHANGED)
            if len(glasses) == 0:
                print("the image is not inplaced")
            else:
                for face in faces:
                    # Create landmark object
                    landmarks = PREDICTOR(image = gray, box = face)
                    # insert bgr into img at desired location and insert mask into black image
                    x1 = int(landmarks.part(40).x)
                    x2 = int(landmarks.part(47).x)
                    y1 = int(landmarks.part(40).y)
                    y2 = int(landmarks.part(47).y)

                    d = abs(x1 - x2)
                    rows, cols = glasses.shape[0], glasses.shape[1]

                    l = abs(y1 - y2)

                    y3 = y1 - y2
                    degree = 0
                    if y3 >= 0:
                        degree = (360 + math.degrees(math.atan2(l, d))) % 360
                    else:
                        degree = -(360 + math.degrees(math.atan2(l, d))) % 360

                    ratio = d * 4 / cols  # cols/3*ratio = d
                    dim = (int(cols * ratio), int(rows * ratio))
                    glasses = cv2.resize(glasses, dim, interpolation = cv2.INTER_AREA)
                    glasses = imutils.rotate(glasses, degree)

                    face_center_y = int((y1 + y2) / 2)
                    face_center_x = int((x1 + x2) / 2)
                    rows, cols = glasses.shape[0], glasses.shape[1]

                    x_offset = face_center_x - int(cols / 2)
                    y_offset = face_center_y - int(rows / 2)

                    for i in range(x_offset, x_offset + cols):
                        for j in range(y_offset, y_offset + rows):
                            if i > 0 and i < input_.shape[1] and j > 0 and j < input_.shape[0] and \
                                    glasses[j - y_offset][i - x_offset][3] != 0:
                                output[j][i][0] = glasses[j - y_offset][i - x_offset][0]
                                output[j][i][1] = glasses[j - y_offset][i - x_offset][1]
                                output[j][i][2] = glasses[j - y_offset][i - x_offset][2]
    elif gui_class.if_clown:
        if len(faces) == 0:
            print("There is no face detected!")
        else:
            glasses = cv2.imread("clown.png", cv2.IMREAD_UNCHANGED)
            if len(glasses) == 0:
                print("the image is not inplaced")
            else:
                for face in faces:
                    # Create landmark object
                    landmarks = PREDICTOR(image = gray, box = face)
                    # insert bgr into img at desired location and insert mask into black image
                    x1 = int(landmarks.part(48).x)
                    x2 = int(landmarks.part(54).x)
                    y1 = int(landmarks.part(48).y)
                    y2 = int(landmarks.part(54).y)

                    d = abs(x1 - x2)
                    rows, cols = glasses.shape[0], glasses.shape[1]

                    l = abs(y1 - y2)

                    y3 = y1 - y2
                    degree = 0
                    if y3 >= 0:
                        degree = (360 + math.degrees(math.atan2(l, d))) % 360
                    else:
                        degree = -(360 + math.degrees(math.atan2(l, d))) % 360
                    print(y3, degree)

                    ratio = d * 3 / cols  # cols/3*ratio = d
                    dim = (int(cols * ratio), int(rows * ratio))
                    glasses = cv2.resize(glasses, dim, interpolation = cv2.INTER_AREA)
                    # glasses = cv2.rotate(glasses, cv2.ROTATE_23_CLOCKWISE)
                    glasses = imutils.rotate(glasses, degree)

                    face_center_y = int((y1 + y2) / 2)
                    face_center_x = int((x1 + x2) / 2)
                    rows, cols = glasses.shape[0], glasses.shape[1]

                    x_offset = face_center_x - int(cols / 2)
                    y_offset = face_center_y - int(rows / 2)

                    for i in range(x_offset, x_offset + cols):
                        for j in range(y_offset, y_offset + rows):
                            if i > 0 and i < input_.shape[1] and j > 0 and j < input_.shape[0] and \
                                    glasses[j - y_offset][i - x_offset][3] != 0:
                                # print(i, j)
                                output[j][i][0] = glasses[j - y_offset][i - x_offset][0]
                                output[j][i][1] = glasses[j - y_offset][i - x_offset][1]
                                output[j][i][2] = glasses[j - y_offset][i - x_offset][2]
    return output
