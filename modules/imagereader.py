
import cv2
import os
import random
from imutils import paths


class ImageReader:
    """ This class is responsible for reading the
    dataset images """

    def __init__(self, dirname, samplesize=60):
        print("\n[INFO] Instanciating ImageReader\n")
        self.dirname = dirname.strip("/")
        self.full_path = "output/" + dirname.strip("/")
        self.samplesize = samplesize

    def load_data(self):
        imagePaths = list(paths.list_images(self.dirname))
        random.shuffle(imagePaths)

        # TODO
        # USE GLOB TO GET LIST OF NAMES.
        # RANDOMIZE BY SAMPLESIZE
        # glob.glob("output/*")

        data = []
        labels = []

        for path in imagePaths:
            # Load image and process it
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (48,62))

            # Append to lists
            data.append(face)
            labels.append(path.split("/")[-2])


        return (data, labels)
