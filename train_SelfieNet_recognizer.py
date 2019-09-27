# USAGE

# import the necessary packages
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import np_utils
from modules.nn import SelfieNet
from modules import ImageReader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import pickle

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="output", help="path to dir containing dataset directories")
ap.add_argument("-n", "--samplesize", type=int, default=60, help="maximum sample size for each face")
ap.add_argument("-m", "--models",default="models", help="directory name for SelfieNet model")
ap.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs in training")
ap.add_argument("-s", "--size", type=int, default=46, help="Image dimension fed into SelfieNet")
args = vars(ap.parse_args())


# OOP. Instanciate objects (incl. ocal Binary Patterns Histogram)
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
io = ImageReader(args["dataset"], int(args["samplesize"]))
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


# Load the data from disk. Split to training and testing sets.
(data, labels) = io.load_data(w=args["size"], h=args["size"], to_array=True, applylbp=False)
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

# TODO! Try out different optimizers such as Adam
opt = SGD(lr=0.01, decay=0.01/args["epochs"], momentum=0.9, nesterov=True)


model = SelfieNet.build(width=args["size"], height=args["size"], depth=1, classes=count_labels)
model.compile(loss="categorical_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY,
	validation_data=(testX, testY), batch_size=64,
	epochs=args["epochs"], verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["models"] + "/SelfieNet.hdf5")
with open(args["models"]+"/labels_SelfieNet.pickle", "wb") as f:
    pickle.dump(le.classes_, f, pickle.HIGHEST_PROTOCOL)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, args["epochs"]), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, args["epochs"]), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, args["epochs"]), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
