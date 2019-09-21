import cv2
import os
import numpy as np
from skimage import feature

def get_face_coords(detector, image):
    # HAAR Cascade detection in Open CV. Keeps only the largest.
    faceRects = detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=9,
                minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
    return faceRects

def keep_largest(faceRects):
    (x, y, w, h) = max(faceRects, key=lambda b:(b[2] * b[3]))
    return (x, y, w, h)
    
def crop_face(frame, x, y, w, h):
    face = frame[y:y + h, x:x + w].copy(order="C")
    return face

def create_dir(dir):
    try:
        os.makedirs(dir)
        print("[INFO] Directory '" + dir + "' created")
    except FileExistsError:
        print("[INFO] Directory '" + dir + "' exists.")
        
def resize_ellipse_face(face, width=62, height=62, lbp=False):
    # Resize the image to the desired dimensions
    (w, h) = (width, height)
    
    if(lbp==True):
        face = feature.local_binary_pattern(face, 26, 1, method="default")
    
    face = cv2.resize(face, (w,h))
    
    # Generate a mask
    mask = np.zeros((h,w), dtype="uint8")
    # cv2.circle(mask, (w//2, h//2), min(w,h)//2, 255, -1)
    cv2.ellipse(mask,(w//2,h//2),(int(w/2*0.7),h//2),0,0,360,255,-1)
    face = cv2.bitwise_and(face, face, mask=mask)
    return face
    