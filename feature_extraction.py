import cv2
import numpy as np
import os
import sqlite_manager


def get_images_and_labels(name, cap):
    TRAINING_SET_COUNT = 500
    images = []
    labels = []
    counter = 0
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    value = 0
    max_val = sqlite_manager.get_max_value()
    if max_val == None:
        max_val = 0
    id = sqlite_manager.name_present(name)
    if not id:
        sqlite_manager.add_new_entry(name)
        value = max_val+1
    else:
        value = id

    # imgs = []
    # for filename in os.listdir('Sriju'):
    #     img = cv2.imread(os.path.join('Sriju', filename))
    #     if img is not None:
    #         imgs.append(img)
    for x in range(0, TRAINING_SET_COUNT):
        # for frame in imgs:
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dir = name
        cv2.imwrite(dir + '/' + dir + str(counter) + '.png', frame)
        counter += 1
        image = np.array(gray, 'uint8')
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )

        # print("Found {0} faces!".format(len(faces)))
        if len(faces) > 1:
            print("Only " + name + " should stand before the camera. Others please move out of the frame")
            continue
        print("Training Count " + str(counter))
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            images.append(image[x:x + w, y:y + h])
            labels.append(value)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
    return images, labels
