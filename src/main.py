# This is a sample Python script.

import cv2
import numpy
import dlib


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

COLOUR_CORRECT_BLUR_FRAC = 0.6

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

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
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def main():

    if_save = False
    if_swich = False

    detector = dlib.get_frontal_face_detector()
    # Load the predictor
    predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
    # read the image
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # Convert image into grayscale
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        # Use detector to find landmarks
        faces = detector(gray)

        if if_save:
            print(len(faces))
            for face in faces:
                x1 = face.left()  # left point
                y1 = face.top()  # top point
                x2 = face.right()  # right point
                y2 = face.bottom()  # bottom point
                # Create landmark object
                landmarks = predictor(image=gray, box=face)

                
                # Loop through all the points
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    # Draw a circle
                    cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
        if if_swich:
            if len(faces) < 2:
                if_swich = False
                print("not enough faces")
                continue
            
            landmarks1 = numpy.matrix([[p.x, p.y] for p in predictor(frame, faces[0]).parts()])
            landmarks2 = numpy.matrix([[p.x, p.y] for p in predictor(frame, faces[1]).parts()])


            M1 = transformation_from_points(landmarks1[ALIGN_POINTS],
                               landmarks2[ALIGN_POINTS])

            M2 = transformation_from_points(landmarks2[ALIGN_POINTS],
                               landmarks1[ALIGN_POINTS])

            mask1 = get_face_mask(frame, landmarks2)
            mask2 = get_face_mask(frame, landmarks1)
            warped_mask1 = warp_im(mask1, M1, frame.shape)
            warped_mask2 = warp_im(mask2, M2, frame.shape)
            combined_mask1 = numpy.max([get_face_mask(frame, landmarks1), warped_mask1],
                          axis=0)
            combined_mask2 = numpy.max([get_face_mask(frame, landmarks2), warped_mask2],
                          axis=0)
            warped_im1 = warp_im(frame, M1, frame.shape)
            warped_im2 = warp_im(frame, M2, frame.shape)
            warped_corrected_im1 = correct_colours(frame, warped_im1, landmarks1)
            warped_corrected_im2 = correct_colours(frame, warped_im2, landmarks2)

            output_im = frame * (1.0 - combined_mask1) + warped_corrected_im1 * combined_mask1
            # output_im = frame * (1.0  - combined_mask1)*(1.0- combined_mask2) + warped_corrected_im2 * combined_mask2 + warped_corrected_im1 * combined_mask1
            output_im = output_im * (1.0 - combined_mask2) + warped_corrected_im2 * combined_mask2

            cv2.imwrite('output.jpg', output_im)

            demo_image = output_im.astype(numpy.uint8)
            cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Display', demo_image)

            print("sucess")
            if_swich = False
            
        
        
        cv2.imshow(winname="Face", mat=frame)
        # Close all windows
        k = cv2.waitKey(10)
        if k == 113:
            cv2.destroyAllWindows()
            break
        if k == 115:
            print("pressed s")
            if if_save == True:
                if_save = False
            else:
                if_save = True
        if k == 119:
            print("pressed w")
            if if_swich == True:
                if_swich = False
            else:
                if_swich = True

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
