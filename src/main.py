# This is a sample Python script.
from time import time

import cv2
import numpy
import dlib
import numpy as np

WINDOWS_WID = 480
WINDOWS_HEI = 360

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

COLOUR_CORRECT_BLUR_FRAC = 0.5

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
# OVERLAY_POINTS = [
#     LEFT_EYE_POINTS + RIGHT_EYE_POINTS,
#     NOSE_POINTS + MOUTH_POINTS,
# ]
OVERLAY_POINTS = [list(range(0, 68))]

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)


def draw_convex_hull(img, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color = color)


def correct_colours(img1, img2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis = 0) -
        numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis = 0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

    return (img2.astype(numpy.float64) * img1_blur.astype(numpy.float64) /
            img2_blur.astype(numpy.float64))


def warp_img(im, M, dshape):
    output_img = numpy.zeros(dshape, dtype = im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst = output_img,
                   borderMode = cv2.BORDER_TRANSPARENT,
                   flags = cv2.WARP_INVERSE_MAP)
    return output_img


def get_face_mask(img, landmarks):
    # im = numpy.zeros(img.shape[:2], dtype = numpy.float64)
    # print(im.shape)
    #
    # for group in OVERLAY_POINTS:
    #     draw_convex_hull(im,
    #                      landmarks[group],
    #                      color = 1)
    #
    # img = numpy.array([im, im, im]).transpose((1, 2, 0))
    # cv2.imshow('tmp', img)
    # cv2.waitKey()
    mask = np.zeros_like(img)
    contex_hull = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, contex_hull, (255, 255, 255))

    # img = (cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    # img = cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return mask


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation.
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis = 0)
    c2 = numpy.mean(points2, axis = 0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) whereas our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def main():
    is_shown_feats = False
    is_face_exchange = False

    detector = dlib.get_frontal_face_detector()
    # Load the predictor
    predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")
    # read the image
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOWS_WID)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOWS_HEI)
    while True:
        # ret, frame = cap.read()
        # exchanged_faces = cap.read()
        frame = cv2.imread('tmp.png')
        frame = cv2.resize(frame, (int(frame.shape[1] * 20 / 100), int(frame.shape[0] * 20 / 100)))
        exchanged_faces = frame.copy()
        # Convert image into grayscale
        gray = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2GRAY)
        # Use detector to find landmarks
        faces = detector(gray)

        if is_shown_feats:
            print(len(faces))
            for face in faces:
                # Create landmark object
                landmarks = predictor(image = gray, box = face)
                # Loop through all the points
                c = 0
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.putText(img = frame, text = str(c), org = (x, y), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = 0.3, color = (0, 0, 255))
                    c += 1
        if is_face_exchange:
            if len(faces) < 2:
                exchanged_faces = frame
                print("Not enough faces")
            else:
                # TODO: May change to exchange multiple faces randomly
                landmarks1 = numpy.matrix([[p.x, p.y] for p in predictor(frame, faces[0]).parts()])
                landmarks2 = numpy.matrix([[p.x, p.y] for p in predictor(frame, faces[1]).parts()])

                M1 = transformation_from_points(landmarks1, landmarks2)
                M2 = transformation_from_points(landmarks2, landmarks1)

                mask1 = get_face_mask(gray, landmarks2)
                mask2 = get_face_mask(frame, landmarks1)
                print(mask1.shape)
                cv2.imshow('mask1', mask1)
                cv2.imshow('mask2', mask2)

                warped_mask1 = warp_img(mask1, M1, frame.shape)
                warped_mask2 = warp_img(mask2, M2, frame.shape)
                cv2.imshow('warped_mask1', warped_mask1)
                cv2.imshow('warped_mask2', warped_mask2)

                combined_mask1 = numpy.max([get_face_mask(frame, landmarks1), warped_mask1], axis = 0)
                combined_mask2 = numpy.max([get_face_mask(frame, landmarks2), warped_mask2], axis = 0)
                cv2.imshow('combined_mask1', combined_mask1)
                cv2.imshow('combined_mask2', combined_mask2)

                warped_img1 = warp_img(frame, M1, frame.shape)
                warped_img2 = warp_img(frame, M2, frame.shape)
                cv2.imshow('warped_img1', warped_img1)
                cv2.imshow('warped_img2', warped_img2)

                # warped_corrected_img1 = correct_colours(frame, warped_img1, landmarks1)
                # warped_corrected_img2 = correct_colours(frame, warped_img2, landmarks2)

                # output_img = frame * (1.0 - combined_mask1) + warped_corrected_img1 * combined_mask1
                output_img = frame * (1.0 - combined_mask1) + warped_img1 * combined_mask1
                output_img = cv2.bitwise_not(frame, frame, mask = mask1)
                # output_img = output_img * (1.0 - combined_mask2) + warped_corrected_img2 * combined_mask2

                exchanged_faces = output_img.astype(numpy.uint8)
                # cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
                # print("success")
        else:
            exchanged_faces = frame

        # cv2.namedWindow('Display')
        # cv2.resizeWindow('Display', WINDOWS_WID, WINDOWS_HEI)
        cv2.imshow('Display', exchanged_faces)
        # cv2.namedWindow('Face')
        # cv2.resizeWindow('Face', WINDOWS_WID, WINDOWS_HEI)
        cv2.imshow('Face', frame)

        # Close all windows
        k = cv2.waitKey(10)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        if k == ord('s'):
            print("pressed s")
            if is_shown_feats:
                is_shown_feats = False
            else:
                is_shown_feats = True
        if k == ord('w'):
            print("pressed w")
            if is_face_exchange:
                is_face_exchange = False
            else:
                is_face_exchange = True


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
