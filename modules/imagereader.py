
import cv2
import os
import random
import glob
from imutils import paths
from .haar_helpers import resize_ellipse_face
from keras.preprocessing.image import img_to_array


class ImageReader:
    """ This class is responsible for reading the
    dataset images """

    def __init__(self, dirname, samplesize):
        print("\n[INFO] Instanciating ImageReader\n")
        self.dirname = dirname.strip("/")
        self.samplesize = samplesize

    def load_data(self, w=62, h=62, to_array=False):
        imagePaths = self._samplegetter()
        random.shuffle(imagePaths)
        
        data = []
        labels = []

        for path in imagePaths:
            # Load image and process it
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = resize_ellipse_face(gray, width=w, height=h)
            
            # Convert to array
            if(to_array):
                face = img_to_array(face)
            
            # Append to lists
            data.append(face)
            labels.append(path.split("/")[-2])


        return (data, labels)


    def _samplegetter(self):
        folders = glob.glob(self.dirname + "/*")
                
        imagePaths = []
        
        for folder in folders:
            name = folder[folder.rfind("/") + 1:]
            selected = list(paths.list_images("output/" + str(name)))
            selected = selected[:self.samplesize]
            for s in selected:
                imagePaths.append(s)
                
        return imagePaths