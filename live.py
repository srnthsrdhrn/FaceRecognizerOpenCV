# This script will detect faces via your webcam.
# Tested with OpenCV3
import threading

import cv2
import numpy as np
import os
import sqlite_manager
from feature_extraction import get_images_and_labels

cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
counter = 0
dir = "default"
name = ""
recognizer = cv2.face.createLBPHFaceRecognizer()
flag = False

images = []
labels = []
try:
    recognizer.load('face_recognizer.yaml')
    flag = True
except:
    print("No Recognizer file Exist")


def wait_for_user_input():
    name = input("Enter Your Name: ")
    cv2.destroyAllWindows()
    images, labels = get_images_and_labels(name, cap)
    recognizer.update(images, np.array(labels))
    recognizer.save('face_recognizer.yaml')
    global flag
    flag = True


def predict():
    counter = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if counter == 0:
        #     dir = input("Enter Name")
        #     os.system('mkdir ' + dir)
        #     cv2.imwrite(dir + '/'+dir + str(counter) + '.png', frame)
        counter += 1
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )

        # print("Found {0} faces!".format(len(faces)))

        # Draw a rectangle around the faces
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = "Couldn't Identify"
        for (x, y, w, h) in faces:
            try:
                predict_image = np.array(gray)
                if flag:
                    value = recognizer.predict(predict_image[x:x + w, y:y + h])
                    name = sqlite_manager.get_name(value)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(name), (x, y - 10), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            except:
                print("OpenCV Exception")
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # if cv2.waitKey(1) and 0xFF == ord('t'):
        #     cv2.destroyAllWindows()
        #     wait_for_user_input()
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break  # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


a = input("Enter T to Train, and P to Predict")
if a.upper() == 'T':
    wait_for_user_input()
elif a.upper() == 'P':
    predict()
else:
    print("Improper Choice, Quitting")

# fps = cap.get(cv2.CAP_PROP_FPS)
# print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
#
#
# # Number of frames to capture
# num_frames = 120
#
# print("Capturing {0} frames".format(num_frames))
#
# # Start time
# start = time.time()
#
# # Grab a few frames
# for i in range(0, num_frames):
#     ret, frame = cap.read()
#
# # End time
# end = time.time()
#
# # Time elapsed
# seconds = end - start
# print("Time taken : {0} seconds".format(seconds))
#
# # Calculate frames per second
# fps = num_frames / seconds
# print("Estimated frames per second : {0}".format(fps))
