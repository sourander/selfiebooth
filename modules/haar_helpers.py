import cv2

def get_face_coords(detector, image):
    # HAAR Cascade detection in Open CV. Keeps only the largest.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7,
                minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
    return faceRects

def keep_largest(faceRects):
    (x, y, w, h) = max(faceRects, key=lambda b:(b[2] * b[3]))
    return (x, y, w, h)
    
def crop_face(frame, x, y, w, h):
    face = frame[y:y + h, x:x + w].copy(order="C")
    return face