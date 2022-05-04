"""
CS5330 - Pattern Recognition & Computer Vision
Project Title: Face detection and filter application
April 18, 2022
Team: Sida Zhang, Xichen Liu, Xiang Wang, Hongyu Wan

Description: ain function of application
"""
__author__ = "Sida Zhang, Hongyu Wan, Xiang Wang, Xichen Liu"

import tkinter
import gui_class


# Create a window and pass it to the Application object
def main():
    gui_class.App(tkinter.Tk(), "Face Detection and its Applications")


if __name__ == '__main__':
    main()
