# USAGE

# import the necessary packages
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import np_utils
from modules.nn import SelfieNet
from modules import ImageReader
from modules import Conf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import pickle

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="Path to config e.g. conf/selfienet.conf")
args = vars(ap.parse_args())

# Load JSON config file
conf = Conf(args["conf"])


# OOP. Instanciate objects (incl. ocal Binary Patterns Histogram)
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
io = ImageReader(conf["datasetpath"], int(conf["samplesize"]))
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


# Load the data from disk. Split to training and testing sets.
(data, labels) = io.load_data(w=conf["size"], h=conf["size"], to_array=True, applylbp=conf["lbp"])
print("[INFO] Size of a single image file is:", data[0].shape)
print("[INFO] The amount of items in dataset:", len(labels))

# scale data to the range of [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
count_labels = len(le.classes_)
labels = np_utils.to_categorical(le.transform(labels), count_labels)
print("[INFO] Unique labels: {}".format(count_labels))

# partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels)


# initialize the optimizer and model
print("[INFO] compiling model...")


model = SelfieNet.build(width=conf["size"], height=conf["size"], depth=1, classes=count_labels)
model.compile(loss="categorical_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
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
with open(conf["modelspath"]+"/labels_SelfieNet.pickle", "wb") as f:
    pickle.dump(le.classes_, f, pickle.HIGHEST_PROTOCOL)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, conf["epochs"]), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, conf["epochs"]), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, conf["epochs"]), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, conf["epochs"]), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
