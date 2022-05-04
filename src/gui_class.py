"""
CS5330 - Pattern Recognition & Computer Vision
Project Title: Face detection and filter application
April 18, 2022
Team: Sida Zhang, Xichen Liu, Xiang Wang, Hongyu Wan

Description: Classes used in our GUI including: Tkinter GUI,
OpenCV real-time feeds, and printing terminal text in GUI
"""
__author__ = "Sida Zhang, Hongyu Wan, Xiang Wang, Xichen Liu"

import csv
import sys
import cv2
import time
import torch
import tkinter
import PIL.Image
import numpy as np
import PIL.ImageTk
from functools import partial
from torch.utils.data import DataLoader
import model_build
import gui_functions
import processing_functions

torch.manual_seed(888)
global if_glass
global if_clown
if_glass = False
if_clown = False

class App:
    def __init__(self, window, window_title, video_source = 0):
        self.mode = 10
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Button
        self.btn_matching = tkinter.Button(window, text = "Matching", width = 50, command = self.getmatching_image)
        self.btn_matching.pack(anchor = tkinter.N, side = "bottom", expand = True)
        self.btn_savecsv = tkinter.Button(window, text = "Create embedding space to database", width = 50,
                                          command = self.get_traintocsv)
        self.btn_savecsv.pack(anchor = tkinter.N, side = "bottom", expand = True)

        # check button for Face Swap
        self.list_itmes = tkinter.StringVar()
        self.list_itmes.set(('glass', 'clown'))
        self.filterchoose = tkinter.Listbox(window, listvariable = self.list_itmes, width = 30, height = 3,
                                            justify = "center")
        self.filterchoose.pack(side = "bottom")

        self.btn_filter = tkinter.Button(window, text = "Face Modifications", width = 50,
                                         command = partial(self.setmode_filter, 3))
        self.btn_filter.pack(anchor = tkinter.N, side = "bottom")

        self.btn_exchange = tkinter.Button(window, text = "Face Swap", width = 50, command = partial(self.setmode, 2))
        self.btn_exchange.pack(anchor = tkinter.N, side = "bottom")

        self.btn_detect = tkinter.Button(window, text = "Face Detect", width = 50, command = partial(self.setmode, 1))
        self.btn_detect.pack(anchor = tkinter.N, side = "bottom")

        self.button0 = tkinter.Button(window, text = "Grayscale", width = 50, command = partial(self.setmode, 0))
        self.button0.pack(anchor = tkinter.N, side = "bottom")

        self.btn_snapshot = tkinter.Button(window, text = "Snapshot", width = 50, command = self.snapshot)
        self.btn_snapshot.pack(anchor = tkinter.N, side = "bottom", expand = True)

        self.vid = MyVideoCapture(self.video_source)

        # Text
        self.Text = tkinter.Text(window, wrap = 'word', width = 120, height = 5)
        self.Text.pack(fill = "both", anchor = tkinter.SW, side = "bottom")
        self.Text.tag_configure('stderr', foreground = '#b22222')

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = 950, height = 300)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()
        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("../" + "data/test/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg",
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

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
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(50, 0, image = self.photo, anchor = tkinter.NW)
            self.photo_output = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(output))
            self.canvas.create_image(500, 0, image = self.photo_output, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)
        sys.stdout = TextRedirector(self.Text, 'stdout')
        sys.stderr = TextRedirector(self.Text, 'stderr')

    # Create a embedding space database for celebrity image datasets
    def get_traintocsv(self):
        print('start Training')
        model_build.generate_csv('image', 'image_info.csv')

        network = model_build.MyNetwork()
        network.eval()

        cele_faces = model_build.CustomizedDataset(annotations_file = '../data/image_info.csv',
                                                   img_dir = '../data/image')
        cele_faces_loader = DataLoader(dataset = cele_faces,
                                       batch_size = 100,
                                       shuffle = False,
                                       num_workers = 0)

        results, targets = processing_functions.build_embedding_space(network, cele_faces_loader)
        with open('../csv/results.csv', "w", newline = '') as f:
            writer = csv.writer(f)
            for row in results:
                rows = row.cpu().detach().numpy().tolist()
                writer.writerow(rows)
        with open('../csv/targets.csv', "w", newline = '') as f:
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
        with open('../csv/results.csv', "r", newline = '') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = np.array(data)
            data = data.astype(float)
            for line in data:
                results.append(line)

        # read image labels from database
        with open('../csv/targets.csv', "r", newline = '') as f:
            reader = csv.reader(f)
            for row in reader:
                str = ''.join(row)
                targets.append(str)

        network = model_build.MyNetwork()
        network.eval()

        # load saved current image
        test_face = model_build.CustomizedDataset(annotations_file = '../data/test_info.csv',
                                                  img_dir = '../data/test')
        test_face_loader = DataLoader(dataset = test_face,
                                      batch_size = 1,
                                      shuffle = False)
        # create embedding space for current image
        results_t, targets_t = processing_functions.build_embedding_space(network, test_face_loader)

        img = cv2.imread(processing_functions.nn(results, targets, results_t[0]))
        cv2.imshow('tmp', img)


# Class using to capture print to GUI text area
class TextRedirector(object):
    def __init__(self, widget, tag = 'stdout'):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state = 'normal')
        self.widget.insert(tkinter.END, str, (self.tag,))  # (self.tag,) 是设置配置
        self.widget.see(tkinter.END)
        self.widget.configure(state = 'disabled')


# Class using opencv to capture stream video from camera and resize
class MyVideoCapture:
    # initialize
    def __init__(self, video_source = 0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # get frame from stream video
    def get_frame(self):
        ret = None
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (400, 300))
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return ret, None

    # output interface with GUI
    def get_output(self, mode):
        ret = None
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (400, 300))
            if mode == 0:
                output = gui_functions.get_gray(frame)
            elif mode == 1:
                output, faces = gui_functions.get_facedetect(frame)
            elif mode == 2:
                output_0, faces = gui_functions.get_facedetect_nodraw(frame)
                output = gui_functions.get_exchange_face(frame, faces)
            elif mode == 3:
                output_0, faces = gui_functions.get_facedetect_nodraw(frame)
                output = gui_functions.get_filtered(frame, faces)
            else:
                output = frame
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return ret, None

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
