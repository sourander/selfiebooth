# import the necessary packages

from imutils.video import WebcamVideoStream
from imutils.video import FPS
from modules import Conf
from modules import ImageDataHandler
from keras.models import load_model
import numpy as np
import argparse
import imutils
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
idh = ImageDataHandler(bdir=idhconf["baseDir"], d=detector)


# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(conf["model"])


# Import labels
with open(conf["labelsfile"], 'rb') as f:
    labels = pickle.load(f)


# initialize the camra
camera = WebcamVideoStream(src=0).start()

if conf["fps"]:
    fps = FPS().start()


while True:
    # grab the current frame
    frame = camera.read()


    # resize the frame, convert the frame to grayscale, and detect faces in the frame
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # HAAR Cascade detection in Open CV. Return all found rectangles.
    faceRects = idh.get_face_coords(gray)
    
    # loop over the face bounding boxes
    for (i, (x, y, w, h)) in enumerate(faceRects):
        # Crop face from colored 'frame'
        face = gray[y:y + h, x:x + w].copy(order="C")
        
        # Resize to size set in config file
        face = cv2.resize(face, (conf["size"],conf["size"]))
        
        # Perform LBP before predicting if set in config
        if conf["lbp"]:
            face = idh.lbp(face, conf["lbp_points"], conf["lbp_radius"])
        
        # Draw ellipse mask to cover nonimportant background
        face = idh.ellipsemask(face, conf["size"], conf["size"])
        
        # Change from 0-255 to 0-1
        face = np.array(face, dtype="float") / 255.0
        
        # Reformat to e.g. (1, 64, 64, 1) since Keras is expecting 4 dimensions
        face = face.reshape((-1, conf["size"], conf["size"], 1))
         
        # Predict the face
        predictions = model.predict(face, batch_size=32)
        preds = predictions.argmax(axis=1)
        confidence = predictions.max()
            
        # Generate info to be printed for the user
        printinfo = "{}: {:.2f}".format(labels[preds[0]], confidence)
        
        # Add info to frame
        cv2.putText(frame, printinfo, (x, y - 20), font, 0.75, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
 
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

