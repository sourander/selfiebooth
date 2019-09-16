# USAGE
# python selfie-generator.py

# Press SPACEBAR to toggle RECORD ON/OFF. 
# Red rectangle means that RECORD is on.
# Press Q to quit.

# import the necessary packages
import argparse
import cv2
import imutils
from modules.haar_helpers import get_face_coords
from modules.haar_helpers import keep_largest
from modules.haar_helpers import crop_face
from modules import ImageWriter


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", 
    required=True, help="Name of the output folder.")
args = vars(ap.parse_args())

       
# OOP. Instanciate objects.
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
io = ImageWriter(args["output"])
camera = cv2.VideoCapture(0)


""" THIS IS FOR DEBUGGING """
if(camera.isOpened() == False):
    print("[INFO] Camera has not been found. Enabling debug mode.")
    camera = cv2.VideoCapture("test.mp4")


# Initialize variables for capture mode toggling
color, line, capturemode = (0, 255, 0), 1, False


# Main loop begins
# Exit by pressing "Q"
while(True):
        
    # Capture a frame, exit if not available.
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=500)


    # HAAR Cascade detection in Open CV. Return all found rectangles.
    faceRects = get_face_coords(detector, frame)


    # Continue only if face was found
    if (len(faceRects) > 0):
        # Discard other rectangles except the largest
        (x, y, w, h) = keep_largest(faceRects)
        face = crop_face(frame, x, y, w, h)
        
        # Save a file if REC button has been pressed
        if(capturemode):
            io.writefile(face)
    
    
    # Display the image to the user, whether a face was found or not.
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, line)
    cv2.imshow("Current frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    
    # Keyboard controls. For guide, check beginning of the document.
    if key == 32:
        if not capturemode:
            capturemode = True
            color, line = (0, 0, 255), 3
            print("Capturemode has been toggled ON")

        # otherwise, back out of capture mode
        else:
            capturemode = False
            color, line = (0, 255, 0), 1
            print("Capturemode has been toggled OFF")
        # if the `q` key is pressed, break from the loop
    elif key == ord("q"):
        break

    
# Perform cleanup
print("\n-------------------------------")
print("[INFO] Total amount of frames of you: {}".format(io.cursor))
print("[INFO] Files are in output/" + args["output"])
camera.release()
cv2.destroyAllWindows()