# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from modules.haar_helpers import create_dir
from modules import ImageReader
from modules import ImageWriter
import numpy as np
import argparse
import imutils
import cv2
import os
import pickle
 
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--dataset", default="output", help="path to dir containing dataset directories")
ap.add_argument("-n", "--samplesize", type=int, default=60, help="maximum sample size for each face")
ap.add_argument("-m", "--models",default="models", help="directory name for LBP model and other model output data")
args = vars(ap.parse_args())


# OOP. Instanciate objects (incl. ocal Binary Patterns Histogram)
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
io = ImageReader(args["dataset"], int(args["samplesize"]))
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


# Load the data from disk. Split to training and testing sets.
(data, labels) = io.load_data()
print("[INFO] Size of a single image file is:", data[0].shape)
print("[INFO] The amount of items in dataset:", len(labels))


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)


# Encode string labels to unique integers
le = LabelEncoder()
le.fit_transform(trainY)
le.fit_transform(testY)


# Train the LBP
recognizer.train(trainX, le.transform(trainY))


# initialize the list of predictions and confidence scores
print("[INFO] gathering predictions...")
predictions = []
confidence = []


# loop over the test data
for i in range(0, len(testX)):
    print("{} of {}".format(str(i), str(len(testX))))
    # classify the face and update the list of predictions and confidence scores
    (prediction, conf) = recognizer.predict(testX[i])
    predictions.append(prediction)
    confidence.append(conf)

 
# show the classification report
print(classification_report(le.transform(testY), predictions,
    target_names=le.classes_))

    
# Add model to a file for later use
print("[INFO] Writing the model to file: ")
create_dir(args["models"])
recognizer.write(args["models"]+"/LBPH.model")
with open(args["models"]+"/labels.pickle", "wb") as f:
    pickle.dump(le.classes_, f, pickle.HIGHEST_PROTOCOL)
