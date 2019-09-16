
import cv2
import os
from imutils import paths

class ImageWriter:
    """ This class is responsible for writing the
    output file so that appends to the folder. No images
    are being written over. """
    
    def __init__(self, dirname):
        print("\n[INFO] Instanciating ImageWriter\n")
        self.dirname = "output/" + dirname.strip("/")
        self._createdir(self.dirname)
        self.cursor = self._get_cursor_location()
        
    def _get_cursor_location(self):
        imagePaths = list(paths.list_images(self.dirname))
        return len(imagePaths) + 1
        
    def _createdir(self, dir):
        try:
            os.makedirs(dir)
            print("[INFO] Directory '" + dir + "' created")
        except FileExistsError:
            print("[INFO] Directory '" + dir + "' exists. Appending")
        
    def _getfilename(self):
        current = self.cursor
        current = str(current).zfill(6) + ".png"
        self.cursor += 1
        return current

    def writefile(self, image):
        """ This method writes a file to the selected
        output directory 
        
        Filename will be "n + 1".png with 6 digits
        
        Example: 000004.png -> 000005.png"""
        
        path = self.dirname + "/" + self._getfilename()
        cv2.imwrite(path, image)