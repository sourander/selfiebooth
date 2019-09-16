import cv2

def get_face_coords(detector, image):
    # HAAR Cascade detection in Open CV. Keeps only the largest.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    (x, y, w, h) = max(faceRects, key=lambda b:(b[2] * b[3]))
    return (x, y, w, h)