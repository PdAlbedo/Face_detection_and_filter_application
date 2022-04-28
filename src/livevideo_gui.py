import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from functools import partial
import dlib

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")


class App:
    def __init__(self, window, window_title, video_source=0):
        self.mode = 10
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Button
        self.button0 = tkinter.Button(window, text="gray", width=50, command=partial(self.setmode, 0))
        self.button0.pack(anchor=tkinter.CENTER, side="bottom")

        self.btn_detect = tkinter.Button(window, text="face_detect", width=50, command=partial(self.setmode, 1))
        self.btn_detect.pack(anchor=tkinter.CENTER, side="bottom")

        self.btn_exchange = tkinter.Button(window, text="exchangeface", width=50, command=partial(self.setmode, 2))
        self.btn_exchange.pack(anchor=tkinter.CENTER, side="bottom")

        self.btn_filter = tkinter.Button(window, text="filter", width=50, command=partial(self.setmode, 3))
        self.btn_filter.pack(anchor=tkinter.CENTER, side="bottom")
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=1600, height=1000)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def setmode(self, number):
        self.mode = number

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        ret, output = self.vid.get_output(self.mode)
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.photo_output = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(output))
            self.canvas.create_image(900, 0, image=self.photo_output, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.mode = 0

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (800, 600))
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def get_output(self, mode):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (800, 600))
            if mode == 0:
                output = get_gray(frame)
            elif mode == 1:
                output = get_facedetect(frame)
            elif mode == 2:
                output = get_exchangeface(frame)
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


def get_gray(input):
    output = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    return output


def get_facedetect(input):
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
                cv2.circle(img=output, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
    return output


def get_exchangeface(input):
    output = input
    return output


def get_filtered(input):
    output = input
    return output


# Create a window and pass it to the Application object
def main():
    App(tkinter.Tk(), "Tkinter and OpenCV")


if __name__ == '__main__':
    main()
