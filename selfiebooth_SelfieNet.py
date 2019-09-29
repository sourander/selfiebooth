# import the necessary packages

from imutils.video import WebcamVideoStream
from imutils.video import FPS
from modules.haar_helpers import get_face_coords
from modules.haar_helpers import crop_face
from modules.haar_helpers import resize_ellipse_face
from modules import Conf
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import pickle
 
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="Path to config e.g. conf/selfienet.conf")
args = vars(ap.parse_args())

# Load JSON config file
conf = Conf(args["conf"])
 
# initialize the HAAR face detector
detector = cv2.CascadeClassifier(conf["haar"])

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(conf["model"])

# Import labels
with open('models/labels_SelfieNet.pickle', 'rb') as f:
    labels = pickle.load(f)


# Choose visual options
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)

# initialize the camra
camera = WebcamVideoStream(src=0).start()

if conf["fps"]:
    fps = FPS().start()

# loop over the frames of the video
while True:
    # grab the current frame
    frame = camera.read()


    # resize the frame, convert the frame to grayscale, and detect faces in the frame
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # HAAR Cascade detection in Open CV. Return all found rectangles.
    faceRects = get_face_coords(detector, gray)
    
    # loop over the face bounding boxes
    for (i, (x, y, w, h)) in enumerate(faceRects):
        # grab the face to predict
        face = crop_face(gray, x, y, w, h)
        face = resize_ellipse_face(face, width=conf["size"], height=conf["size"], lbp=conf["lbp"])
        
        face = np.array(face, dtype="float") / 255.0
        face = face.reshape((-1, conf["size"], conf["size"], 1))
         
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
    
    if conf["fps"]:
        fps.update()
 
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
 
# stop the timer and display FPS information
if conf["fps"]:
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps())) 

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
camera.stop()

