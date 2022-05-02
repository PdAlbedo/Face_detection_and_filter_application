import tkinter
import PIL.Image, PIL.ImageTk
from functools import partial
import dlib
import imutils
import numpy as np
import math
from torch.utils.data import DataLoader
import model_build
import cv2
import torch
import sys
import csv
import time

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

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

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

if_glass = False
if_clown = False
torch.manual_seed(888)

class App:
    def __init__(self, window, window_title, video_source=0):
        self.mode = 10
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source


        # Button

        self.btn_matching = tkinter.Button(window, text="Matching", width=50, command=self.getmatching_image)
        self.btn_matching.pack(anchor=tkinter.N, side="bottom", expand=True)

        self.btn_savecsv= tkinter.Button(window, text="Create embedding space to database", width=50, command=self.get_traintocsv)
        self.btn_savecsv.pack(anchor=tkinter.N, side="bottom", expand=True)

        #check button for Face Swap
        self.list_itmes = tkinter.StringVar()
        self.list_itmes.set(('glass', 'clown'))
        self.filterchoose = tkinter.Listbox(window, listvariable=self.list_itmes, width=30, height=3, justify="center")
        self.filterchoose.pack(side="bottom")

        self.btn_filter = tkinter.Button(window, text="Face Modifications", width=50, command=partial(self.setmode_filter, 3))
        self.btn_filter.pack(anchor=tkinter.N, side="bottom")

        self.btn_exchange = tkinter.Button(window, text="Face Swap", width=50, command=partial(self.setmode, 2))
        self.btn_exchange.pack(anchor=tkinter.N, side="bottom")

        self.btn_detect = tkinter.Button(window, text="Face Detect", width=50, command=partial(self.setmode, 1))
        self.btn_detect.pack(anchor=tkinter.N, side="bottom")

        self.button0 = tkinter.Button(window, text="Grayscale", width=50, command=partial(self.setmode, 0))
        self.button0.pack(anchor=tkinter.N, side="bottom")

        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.N, side="bottom", expand=True)

        self.vid = MyVideoCapture(self.video_source)
        # text
        self.Text = tkinter.Text(window, wrap='word', width=120, height=5)
        self.Text.pack(fill="both", anchor=tkinter.SW, side="bottom")
        self.Text.tag_configure('stderr', foreground='#b22222')
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=950, height=300)

        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()
        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("../"+"data/test/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    def setmode(self, number):
        self.mode = number

    def setmode_filter(self, number):
        select = self.filterchoose.curselection()
        text = self.filterchoose.get(select)
        print(text)
        global if_glass
        global if_clown
        if text == 'glass':
            if_glass = True
            if_clown = False
        elif text == 'clown':
            if_clown = True
            if_glass = False
        self.mode = number

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        ret, output = self.vid.get_output(self.mode)
        # update stream video show on the image
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(50, 0, image=self.photo, anchor=tkinter.NW)
            self.photo_output = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(output))
            self.canvas.create_image(500, 0, image=self.photo_output, anchor=tkinter.NW)
        self.window.after(self.delay, self.update)
        sys.stdout = TextRedirector(self.Text, 'stdout')
        sys.stderr = TextRedirector(self.Text, 'stderr')

    # Create a embedding space database for celebrity image datasets
    def get_traintocsv(self):
        print('start Training')
        model_build.generate_csv('image', 'image_info.csv')

        network = model_build.MyNetwork()
        network.eval()

        cele_faces = model_build.CustomizedDataset(annotations_file='../data/image_info.csv',
                                                   img_dir='../data/image')
        cele_faces_loader = DataLoader(dataset=cele_faces,
                                       batch_size=100,
                                       shuffle=False,
                                       num_workers=0)


        results, targets = build_embedding_space(network, cele_faces_loader)
        with open('../csv/results.csv',"w",newline='') as f:
            writer = csv.writer(f)
            for row in results:
                rows=row.cpu().detach().numpy().tolist()
                writer.writerow(rows)
        with open('../csv/targets.csv',"w",newline='') as f:
            writer = csv.writer(f)
            writer.writerows(targets)

    # matching image for current videostream faces
    def getmatching_image(self):
        # snapshot for current videostream
        self.snapshot()
        print('snapshot finished')
        print('start Matching')
        model_build.generate_csv('test', 'test_info.csv')
        targets = []
        results = []
        # read embedding space results from database
        with open('../csv/results.csv', "r", newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = np.array(data)
            data = data.astype(float)
            for line in data:
                results.append(line)

        # read image labels from database
        with open('../csv/targets.csv', "r", newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                str = ''.join(row)
                targets.append(str)

        network = model_build.MyNetwork()
        network.eval()

        # load saved current image
        test_face = model_build.CustomizedDataset(annotations_file='../data/test_info.csv',
                                                  img_dir='../data/test')
        test_face_loader = DataLoader(dataset=test_face,
                                      batch_size=1,
                                      shuffle=False)
        # create embedding space for current image
        results_t, targets_t = build_embedding_space(network, test_face_loader)


        img = cv2.imread(nn(results, targets, results_t[0]))
        cv2.imshow('tmp', img)

# Class using to capture print to GUI text area
class TextRedirector(object):
    def __init__(self, widget, tag='stdout'):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state='normal')
        self.widget.insert(tkinter.END, str, (self.tag,))    # (self.tag,) 是设置配置
        self.widget.see(tkinter.END)
        self.widget.configure(state='disabled')

# Class using opencv to capture stream video from camera and resize
class MyVideoCapture:
    # initialize
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # get frame from stream video
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (400, 300))
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
    # output interface with GUI
    def get_output(self, mode):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (400, 300))
            if mode == 0:
                output = get_gray(frame)
            elif mode == 1:
                output, faces = get_facedetect(frame)
            elif mode == 2:
                output_0, faces = get_facedetect(frame)
                output = get_exchangeface(frame, faces)
            elif mode == 3:
                output_0, faces = get_facedetect(frame)
                output = get_filtered(frame, faces)
            else:
                output = frame
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Grayscale function
def get_gray(input):
    print('start grayscale')
    output = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    return output

# Facedetect function
def get_facedetect(input):
    print('start face detect')
    output = input
    gray = cv2.cvtColor(src=input, code=cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = DETECTOR(gray)
    for face in faces:

        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            # Create landmark object
            landmarks = PREDICTOR(image=gray, box=face)

            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                # Draw a circle
                cv2.circle(img=output, center=(x, y), radius=1, color=(0, 255, 0), thickness=-1)
    return output, faces

# Face Swap function
def get_exchangeface(input, faces):
    output = input
    if len(faces) < 2:
        print("not enough faces")

    else:
        landmarks1 = np.matrix([[p.x, p.y] for p in PREDICTOR(input, faces[0]).parts()])
        landmarks2 = np.matrix([[p.x, p.y] for p in PREDICTOR(input, faces[1]).parts()])

        M1 = transformation_from_points(landmarks1[ALIGN_POINTS],
                                        landmarks2[ALIGN_POINTS])

        M2 = transformation_from_points(landmarks2[ALIGN_POINTS],
                                        landmarks1[ALIGN_POINTS])

        mask1 = get_face_mask(input, landmarks2)
        mask2 = get_face_mask(input, landmarks1)
        warped_mask1 = warp_im(mask1, M1, input.shape)
        warped_mask2 = warp_im(mask2, M2, input.shape)
        combined_mask1 = np.max([get_face_mask(input, landmarks1), warped_mask1],
                                   axis=0)
        combined_mask2 = np.max([get_face_mask(input, landmarks2), warped_mask2],
                                   axis=0)
        warped_im1 = warp_im(input, M1, input.shape)
        warped_im2 = warp_im(input, M2, input.shape)
        warped_corrected_im1 = correct_colours(input, warped_im1, landmarks1)
        warped_corrected_im2 = correct_colours(input, warped_im2, landmarks2)

        output_im = input * (1.0 - combined_mask1) + warped_corrected_im1 * combined_mask1
        # output_im = frame * (1.0  - combined_mask1)*(1.0- combined_mask2) + warped_corrected_im2 * combined_mask2 + warped_corrected_im1 * combined_mask1
        output_im = output_im * (1.0 - combined_mask2) + warped_corrected_im2 * combined_mask2

        cv2.imwrite('output.jpg', output_im)

        output = output_im.astype(np.uint8)
    return output

# Face Modifications function
def get_filtered(input, faces):
    print('start filter')
    output = input
    gray = cv2.cvtColor(src=input, code=cv2.COLOR_BGR2GRAY)
    if if_glass:
        if len(faces) == 0:
            print("There is no face detected!")
        else:
            glasses = cv2.imread("sun_glasses.png", cv2.IMREAD_UNCHANGED)
            if len(glasses) == 0:
                print("the image is not inplaced")
            else:
                for face in faces:
                    # Create landmark object
                    landmarks = PREDICTOR(image=gray, box=face)
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
                    glasses = cv2.resize(glasses, dim, interpolation=cv2.INTER_AREA)
                    glasses = imutils.rotate(glasses, degree)

                    face_center_y = int((y1 + y2) / 2)
                    face_center_x = int((x1 + x2) / 2)
                    rows, cols = glasses.shape[0], glasses.shape[1]

                    x_offset = face_center_x - int(cols / 2)
                    y_offset = face_center_y - int(rows / 2)

                    for i in range(x_offset, x_offset + cols):
                        for j in range(y_offset, y_offset + rows):
                            if (i > 0 and i < input.shape[1] and j > 0 and j < input.shape[0]
                                    and glasses[j - y_offset][i - x_offset][3] != 0):
                                output[j][i][0] = glasses[j - y_offset][i - x_offset][0]
                                output[j][i][1] = glasses[j - y_offset][i - x_offset][1]
                                output[j][i][2] = glasses[j - y_offset][i - x_offset][2]
    elif if_clown:
        if len(faces) == 0:
            print("There is no face detected!")
        else:
            glasses = cv2.imread("clown.png", cv2.IMREAD_UNCHANGED)
            if len(glasses) == 0:
                print("the image is not inplaced")
            else:
                for face in faces:
                    # Create landmark object
                    landmarks = PREDICTOR(image=gray, box=face)
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
                    glasses = cv2.resize(glasses, dim, interpolation=cv2.INTER_AREA)
                    # glasses = cv2.rotate(glasses, cv2.ROTATE_23_CLOCKWISE)
                    glasses = imutils.rotate(glasses, degree)

                    face_center_y = int((y1 + y2) / 2)
                    face_center_x = int((x1 + x2) / 2)
                    rows, cols = glasses.shape[0], glasses.shape[1]

                    x_offset = face_center_x - int(cols / 2)
                    y_offset = face_center_y - int(rows / 2)

                    for i in range(x_offset, x_offset + cols):
                        for j in range(y_offset, y_offset + rows):
                            if (i > 0 and i < input.shape[1] and j > 0 and j < input.shape[0]
                                    and glasses[j - y_offset][i - x_offset][3] != 0):
                                # print(i, j)
                                output[j][i][0] = glasses[j - y_offset][i - x_offset][0]
                                output[j][i][1] = glasses[j - y_offset][i - x_offset][1]
                                output[j][i][2] = glasses[j - y_offset][i - x_offset][2]
    return output

# draw face area
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

# correct colours for two detected faces
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
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
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

# get face mask area
def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

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
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
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
    a=a.cpu().detach().numpy()
    targets=np.array(targets)
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

# Create a window and pass it to the Application object
def main():
    App(tkinter.Tk(), "Face Detection and its Applications")



if __name__ == '__main__':
    main()
