# USAGE
# python selfie-generator.py

# import the necessary packages
import argparse
import cv2
import imutils
from modules.haar_helpers import get_face_coords
from modules import ImageWriter


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", 
    required=True, help="Name of the output folder.")
args = vars(ap.parse_args())

       
# OOP. Instanciate objects.
detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
io = ImageWriter(args["output"])

image = cv2.imread("test.png")

# HAAR Cascade detection in Open CV. Keeps only the largest.
(x, y, w, h) = get_face_coords(detector, image)
face = image[y:y + h, x:x + w].copy(order="C")

# Handle file output
io.writefile(face)
