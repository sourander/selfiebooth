# import the necessary packages

from modules.haar_helpers import get_face_coords
from modules import ImageReader
import argparse
import imutils
import cv2
import pickle
 
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classifier", default="models/LBPH.model", help="path to the classifier")
ap.add_argument("-t", "--confidence", type=float, default=100.0,
    help="maximum confidence threshold for positive face identification")
args = vars(ap.parse_args())
 
# initialize the HAAR face detector
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# initialize LBPH recognizer using the models/LBPH.model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(args["classifier"])
recognizer.setThreshold(args["confidence"])


# Import labels
with open('models/labels.pickle', 'rb') as f:
    labels = pickle.load(f)

# Choose visual options
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)

# grab a reference to the webcam
camera = cv2.VideoCapture(0)

# For debugging
if(camera.isOpened() == False):
    print("[INFO] Camera has not been found. Enabling debug mode.")
    camera = cv2.VideoCapture("test.mp4")

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
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48,62))
         
        # Predict the face
        (prediction, confidence) = recognizer.predict(face)

        # Fit labels to the label indexes
        if prediction == -1:
            (prediction, confidence) = ("Unknown", 0.0)
        else:
            prediction = labels[prediction]
        printinfo = "{}: {:.2f}".format(prediction, confidence)
        
        # Generate output info

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

