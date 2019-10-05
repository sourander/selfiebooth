# USAGE

# import the necessary packages
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import np_utils
from modules.nn import SelfieNet
from modules import ImageDataHandler
from modules import Conf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import pickle

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="Path to config e.g. conf/selfienet.conf")
ap.add_argument("-i", "--idhconf", default="conf/idhconf.conf",
    help="Not to be changed in normal condition")
args = vars(ap.parse_args())

# Load JSON config file(s) and set font
idhconf = Conf(args["idhconf"])
conf = Conf(args["conf"])
font = cv2.FONT_HERSHEY_SIMPLEX

# initialize the HAAR face detector
detector = cv2.CascadeClassifier(idhconf["haar"])

# Call the ImageDataHandler which performs all HAAR operations
# and all image input and output
idh = ImageDataHandler(bdir=idhconf["baseDir"], d=detector, sample=conf["samplesize"], size=conf["size"])

# Load data and matching labels from output/*dirs*
# Config will determine if Keras preprocessing is applied
# OpenCV works fine with (64,64) input as uint8
# Keras is expecting (1,64,64,1) as float 0...1
(data, labels) = idh.load_data(conf)

# Count unique labels
count_labels = len(set(labels))

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), count_labels)

# Split dataset into training and test sets
(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=0.20, stratify=labels)

print("[INFO] compiling model...")

if conf["network"] == "keras":
    print("[INFO] compiling neural network model...")
    model = SelfieNet.build(width=conf["size"], height=conf["size"], depth=1, classes=count_labels)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("[INFO] training neural network...")
    H = model.fit(trainX, trainY,
                  validation_data=(testX, testY), batch_size=64,
                  epochs=conf["epochs"], verbose=1)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=64)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=[str(x) for x in le.classes_]))

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(conf["modelspath"] + "/SelfieNet.hdf5")
    with open(conf["modelspath"] + "/labels_SelfieNet.pickle", "wb") as f:
        pickle.dump(list(le.classes_), f, pickle.HIGHEST_PROTOCOL)

