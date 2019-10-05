import numpy as np
import cv2
import os
from imutils import paths
from skimage import feature
import glob
import random


class ImageDataHandler:
    def __init__(self, bdir=None, n=None, d=None, sample=80, size=64):
        # Default 'output', comes from idhconfig.conf
        self.base_dir = bdir
        
        # This applies only if argument --name was given; we want to create new files
        self.subjectname = n
        if n is not None:
            self._createdir(os.path.join(self.base_dir, self.subjectname))
            self.cursor = self._get_cursor_location()
        
        # HAAR
        self.detector = d

        # Sample size for training
        self.samplesize = sample

        # Square image size (m*m)
        self.size = size
        
        
    
    def get_face_coords(self, image):
        faceRects = self.detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=9,
                    minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
        return faceRects

    
    def keep_largest(self, faceRects):
        (x, y, w, h) = max(faceRects, key=lambda b:(b[2] * b[3]))
        return (x, y, w, h)

        
    def writefile(self, image):
        """ This method writes a file to the selected
        output directory 
    
        Filename will be "n + 1".png with 6 digits
        
        Example: 000004.png -> 000005.png"""
        
        # Create output/name/ directory
        dir = os.path.join(self.base_dir, self.subjectname)
        
        # Get filename for saving e.g. output/jani/
        path = dir + "/" + self._getfilename()
        cv2.imwrite(path, image)

    def load_data(self, conf):
        # Load data
        image_paths = self._sample_getter()
        random.shuffle(image_paths)

        data = []
        labels = []

        for path in image_paths:
            # Load and process data
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (self.size,self.size))

            if conf["network"] == "keras":
                # Apply Local Binary Patterns if set
                if conf["lbp"] == True:
                    face = self.lbp(face, conf["lbp_points"], conf["lbp_radius"])

                # Apply Keras preprocessing (e.g. float and array convert)
                face = self.keras_preprocess(face, conf["size"])

            # Append to lists
            data.append(face)
            labels.append(path.split("/")[-2])

        if conf["network"] == "keras":
            # Convert list to numpy array
            data = np.array(data, dtype="float")

        return data, labels

    def keras_preprocess(self, face, size):
        # Apply mask
        face = self.ellipsemask(face, size, size)

        # Change from 0-255 to 0-1
        face = np.array(face, dtype="float") / 255.0

        # Reformat to e.g. (64, 64, 1) since Keras is expecting 3 dimensions
        face = face.reshape((size, size, 1))

        return face
        
    def lbp(self, face, no_points, radius):
        face = feature.local_binary_pattern(face, no_points, radius, method="default")
        return face
        
    
    def ellipsemask(self, image, w, h):
        # Generate a mask
        mask = np.zeros((h,w), dtype="uint8")
        cv2.ellipse(mask,(w//2,h//2),(int(w/2*0.7),h//2),0,0,360,255,-1)
        image = cv2.bitwise_and(image, image, mask=mask)
        return image

    def _sample_getter(self):
        # Get n size sample from each folder in output/
        folders = glob.glob(self.base_dir + "/*")

        image_paths = []

        for folder in folders:
            name = folder[folder.rfind("/") + 1:]
            name_path = os.path.join(self.base_dir, name)
            selected = list(paths.list_images(name_path))
            selected_images = selected[:self.samplesize]
            for img in selected_images:
                image_paths.append(img)
        return image_paths


    def _get_cursor_location(self):
        # Count images in /output/name/ and return (n + 1)
        imagePaths = list(paths.list_images(os.path.join(self.base_dir, self.subjectname)))
        return len(imagePaths) + 1
        
        
    def _getfilename(self):
        current = self.cursor
        current = str(current).zfill(6) + ".png"
        self.cursor += 1
        return current
        
        
    def _createdir(self, dir):
        try:
            os.makedirs(dir)
            print("[INFO] Directory '" + dir + "' created")
        except FileExistsError:
            print("[INFO] Directory '" + dir + "' exists. Appending")

