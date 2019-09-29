# USAGE
# python selfie-generator.py

# Press SPACEBAR to toggle RECORD ON/OFF. 
# Red rectangle means that RECORD is on.
# Press Q to quit.

# import the necessary packages
from modules import ImageDataHandler
from modules import Conf
import argparse
import cv2
import imutils



ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="Your name")
ap.add_argument("-c", "--idhconf", default="conf/idhconf.conf", 
    help="Not to be changed in normal condition")
args = vars(ap.parse_args())


# Load JSON config file(s)
idhconf = Conf(args["idhconf"])


# Load CV2 Cascade Classifier
detector = cv2.CascadeClassifier(idhconf["haar"])


# Call the ImageDataHandler which performs all HAAR operations
# and all image input and output
idh = ImageDataHandler(bdir=idhconf["baseDir"], n=args["name"], d=detector)


# Start camera and set REC off by default
camera = cv2.VideoCapture(0)
capturemode = False
color, line = idhconf["green"], idhconf["thinline"]

# Main loop begins
while(True):
        
    # Capture a frame, exit if not available.
    (grabbed, frame) = camera.read()
    if not grabbed:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # HAAR Cascade detection in Open CV. Return all found rectangles.
    faceRects = idh.get_face_coords(gray)


    # Continue only if face was found
    if (len(faceRects) > 0):
        # Discard other rectangles except the largest
        (x, y, w, h) = idh.keep_largest(faceRects)
        
        # Crop face from colored 'frame'
        face = frame[y:y + h, x:x + w].copy(order="C")
        
        # Save a file if REC button has been pressed
        if(capturemode):
            idh.writefile(face)
    
    
        # Display the image to the user, whether a face was found or not.
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, line)
    
    cv2.imshow("Current frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    
    # Keyboard controls. For guide, check beginning of the document.
    if key == 32:
        if not capturemode:
            capturemode = True
            color, line = idhconf["red"], idhconf["thickline"]
            print("Capturemode has been toggled ON")

        # otherwise, back out of capture mode
        else:
            capturemode = False
            color, line = idhconf["green"], idhconf["thinline"]
            print("Capturemode has been toggled OFF")
        # if the `q` key is pressed, break from the loop
    elif key == ord("q"):
        break

    
# Perform cleanup
print("\n-------------------------------")
print("[INFO] Total amount of frames of you: {}".format(idh.cursor))
print("[INFO] Files are in " + idhconf["baseDir"] + "/" + args["name"])
camera.release()
cv2.destroyAllWindows()