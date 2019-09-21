# import the necessary packages

from modules.haar_helpers import get_face_coords
from modules.haar_helpers import crop_face
from modules.haar_helpers import resize_ellipse_face
from modules import ImageReader
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import pickle
 
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="models/SelfieNet.hdf5", help="path to the classifier")
ap.add_argument("-s", "--size", type=int, default=46, help="Image dimension fed into SelfieNet")
args = vars(ap.parse_args())
 
# initialize the HAAR face detector
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# Import labels
with open('models/labels_SelfieNet.pickle', 'rb') as f:
    labels = pickle.load(f)


# Choose visual options
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)

# grab a reference to the webcam
camera = cv2.VideoCapture(0)

# loop over the frames of the video
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if the frame could not be grabbed, then we have reached the end of the video
    if not grabbed:
        break

    # resize the frame, convert the frame to grayscale, and detect faces in the frame
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # HAAR Cascade detection in Open CV. Return all found rectangles.
    faceRects = get_face_coords(detector, gray)
    
    # loop over the face bounding boxes
    for (i, (x, y, w, h)) in enumerate(faceRects):
        # grab the face to predict
        face = crop_face(gray, x, y, w, h)
        face = resize_ellipse_face(face, width=args["size"], height=args["size"], lbp=False)
        
        face = np.array(face, dtype="float") / 255.0
        face = face.reshape((-1, args["size"], args["size"], 1))
         
        # Predict the face
        preds = model.predict(face, batch_size=32).argmax(axis=1)
        confidence = model.predict(face, batch_size=32).max()
        
        # Turn ["'name'"] into name
        name = str(labels[preds].astype(str))[2:-2]
        
        # Generate info to be printed for the user
        printinfo = "{}: {:.2f}".format(name, confidence)
        
        # Add info to frame
        cv2.putText(frame, printinfo, (x, y - 20), font, 0.75, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
 
    # show the frame and record if the user presses a key
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

