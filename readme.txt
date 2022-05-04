# CS5330 - Pattern Recognition & Computer Vision
# Final Project: Face detection and filter application
# Project Title: Face detection and filter application
# April 18, 2022
# Sida Zhang, Xichen Liu, Xiang Wang, Hongyu Wan
# Presentation: https://drive.google.com/file/d/19Dm4UWrC8CCqG4AyaWl9m4RVnM__9esc/view?usp=sharing
# Demo: https://drive.google.com/file/d/17AmMfGcMpK75dKs9108RZokUaeD1YqVq/view
# Enviornment:
     OS: Windows and MacOS
     IDE: PyCharm
     Main Imports: OpenCV, Tkinter, Torch, dlib, torchvision.
# Projectr Structure:
  - Final_Project:
      - csv:
        -   results.csv
        -   targets.csv
      - data:
        -   image:
           - 621 celebrity jpg images   
        -   test: 
           - snapshot location
        -   shape_predictor_68_face_landmarks.dat
      - src
        -   clown.png
        -   sun_glasses.png
        -   gui_class.py
        -   gui_functions.py
        -   livevideo_gui.py (main)
        -   model_build.py
        -   processing_function.py 
      - readme.txt


# Description:
The project will be able to detect the human face and apply some filters or other
operations to the detected faces.

Our final project is designed to be a computer vision-based facial detection app
emphasizing feature extraction and analysis. With live camera inputs, we are able
to detect faces and pull the features for functions including filters and
characteristic transformation. It is a facial recognition app based on python.

In this app, we mainly use live footage as data input, and users are able to use
various filters to enrich their videos. During the live broadcast, it analyzes
the screen, extracts possible facial features, identifies the user’s face and its
expressions, and extracts the user’s facial data. With this data, our app will be
able to match these features to our pre- designed filters and present users with
rich visual effects.
