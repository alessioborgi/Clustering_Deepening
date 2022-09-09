#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 19 08:42:32 2022
@Author:     Alessio Borgi
@Contact :   alessioborgi3@gmail.com
@Filename:   Object_Tracking_Mean_Shift.py
"""

#Importing necessary libraries.
import numpy as np
import cv2 as cv

#Importing the Traffic Video.
cap = cv.VideoCapture('traffic_video.mp4')

#Taking the first frame of the video.
ret, frame = cap.read()

#Setting up the initial location in window. (Here we set up it to the white car initial location).
x, y, width, height = 600, 400, 200, 100

#Setting up the Tracking Window.
track_window = (x, y ,width, height)

#Setting up the Tracking ROI.
roi = frame[y:y+height, x : x+width]

#Changing color space for the ROI.
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

#Applying range Mask.
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255)))

#Computing the Histogram.
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])

#Normalize the retrieved value.
cv.normalize(roi_hist, roi_hist, 0, 255,cv.NORM_MINMAX)

#Setting up Termination Criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 2)
cv.imshow('roi',roi)


#Looping over the Video and Tracking the White Car.
while(True):
    
    #Reading the Frame.
    ret, frame = cap.read()
    
    #Checking whether the frame is returned or not.
    if ret == True:

        #Changing Color Space from BGR to HSV.
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        #Computing the Back-Projection on the ROI in the new Image.
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        
        #Applying MeanShift Clustering in order to get the new location.
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        
        #Obtaining and Drawing the Object Tracking rectangle on the window.
        x,y,w,h = track_window
        final_image = cv.rectangle(frame, (x,y), (x+w, y+h), 255, 3)

        #Showing the Mean-Shift Result.        
        cv.imshow('Object Tracking: Mean-Shift Clustering',final_image)
        
        #Checking whether we are clicking a button for exiting the video.
        k = cv.waitKey(30) & 0xff
        if k == 27 or k == ord('q') or k == ord(' '):
            break
    else:
        break