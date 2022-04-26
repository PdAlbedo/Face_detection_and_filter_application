# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math
import random

import cv2
import numpy as np
import dlib
import point2d


def main():
    detector = dlib.get_frontal_face_detector()
    # Load the predictor
    predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")
    # read the image
    # cap = cv2.VideoCapture(0) TODO
    while True:
        # ret, frame = cap.read() TODO
        frame = cv2.imread("tmp.png")
        frame = cv2.resize(frame, (int(frame.shape[1] * 40 / 100), int(frame.shape[0] * 40 / 100)))
        # Convert image into grayscale
        gray = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2GRAY)
        # Use detector to find landmarks
        faces = detector(gray)
        face_feats = []
        for face in faces:
            face_feat = []
            # Create landmark object
            landmarks = predictor(image = gray, box = face)
            # Loop through all the points
            c = 0
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                face_feat.append((x, y))
                # Mark the num
                cv2.putText(img = frame, text = str(c), org = (x, y), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.3, color = (255, 0, 0))
                c += 1
            face_feats.append(np.array(face_feat))

        if len(face_feats) > 1:
            extracted_faces = []
            hor_lens = []
            ver_lens = []
            idx = []
            for i in range(len(face_feats)):
                # cv2.line(frame, face_feats[0][8], face_feats[0][27], (0, 0, 255), 2)
                # cv2.line(frame, face_feats[1][8], face_feats[1][27], (0, 0, 255), 2)
                # line1Y1 = face_feats[0][8][1]
                # line1X1 = face_feats[0][8][0]
                # line1Y2 = face_feats[0][27][1]
                # line1X2 = face_feats[0][27][0]
                # line2Y1 = face_feats[1][8][1]
                # line2X1 = face_feats[1][8][0]
                # line2Y2 = face_feats[1][27][1]
                # line2X2 = face_feats[1][27][0]
                #
                # #calculate angle between pairs of lines
                # angle1 = math.atan2(line1Y1-line1Y2,line1X1-line1X2)
                # angle2 = math.atan2(line2Y1-line2Y2,line2X1-line2X2)
                # angleDegrees = (angle1-angle2) * 360 / (2 * math.pi)
                # print(angleDegrees + 360)
                # contours, hierarchy = cv2.findContours(face_feats[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                idx.append(i)

                hor_dis = math.dist(face_feats[i][0], face_feats[i][16])
                ver_dis = math.dist(face_feats[i][27], face_feats[i][8])
                hor_lens.append(hor_dis)
                ver_lens.append(ver_dis)

                mask = np.zeros_like(gray)
                contex_hull = cv2.convexHull(face_feats[i])
                cv2.polylines(frame, [contex_hull], True, (0, 0, 255), 2)
                cv2.fillConvexPoly(mask, contex_hull, 255)
                extracted_face = cv2.bitwise_and(frame, frame, mask = mask)
                extracted_faces.append(extracted_face)

            shuffled_idx = idx.copy()
            while shuffled_idx == idx:
                random.shuffle(shuffled_idx)
            print(idx)

            c = 0
            for i in extracted_faces:
                # faces = cv2.bitwise_and(frame, frame, mask = i)
                cv2.imshow(str(c), i)
                c += 1



        # show the image
        cv2.imshow(winname = "Face", mat = frame)
        # Close all windows
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
