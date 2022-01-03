import cv2
from deepface import DeepFace
import numpy as np  #this will be used later in the process
from pathlib import Path

pathlist = Path('.\\data\\age\\50').rglob('*.png')
pictures = []
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    pictures.append([path_in_str, 50])

correct = 0
false = 0
no_face = 0

for i in pictures:
    image = cv2.imread(i[0])

    try:
        analyze = DeepFace.analyze(image, actions=['age'], enforce_detection=True)
        if analyze['age'] > 40 and analyze['age'] < 60:
            correct += 1
        else:
            false += 1
    except:
        no_face += 1

    

print("Correct:", correct)
print("False:", false)
print("No Face Found:", no_face)