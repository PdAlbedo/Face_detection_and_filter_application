import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time


class App:
    def __init__(self, window, window_title, video_source=0):
        self.model = 0
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        #Button
        self.button = tkinter.Button(window, text ="gray",width = 50, command = MyVideoCapture.set_mode(MyVideoCapture,1))
        self.button.pack(anchor=tkinter.CENTER, expand=True)
         # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = 1000, height = 600)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame= self.vid.get_frame()
        ret, output = self.vid.get_output()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.photo_output = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(output))
            self.canvas.create_image(500, 0, image = self.photo_output, anchor = tkinter.NW)

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
             frame =cv2.resize(frame, (400,400))
             if ret:
                # Return a boolean success flag and the current frame converted to BGR
                    return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                return (ret, None)
        else:
            return (ret, None)

    def get_output(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (400, 400))
            output =frame
            print(self.mode)
            if self.mode==1:
                output = get_gray(frame)
            if ret:
               # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def set_mode(self, number):
        self.set_mode() = number



    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def get_gray(input):
    output = cv2.cvtColor(input,cv2.COLOR_BGR2RGB)
    return output


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")