import cv2
from deepface import DeepFace
import numpy as np  #this will be used later in the process
from pathlib import Path

#imgpath = r"C:\Users\Fred\Desktop\Facebank\pics\2204.jpg"
#image = cv2.imread(imgpath)

pathlist = Path('.\\data\\age\\10').rglob('*.png')
pictures = np.array([])
count = 0
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    print("test")
    np.insert(pictures,count,[path_in_str,10])
    count = count + 1
print(pictures)
