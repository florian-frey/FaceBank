#emotion_detection.py
import cv2
from deepface import DeepFace
import numpy as np  #this will be used later in the process

imgpath = r"C:\Users\Fred\Desktop\Facebank\pics\2204.jpg"
image = cv2.imread(imgpath)

analyze = DeepFace.analyze(image,actions=['emotion','age','race'])  #here the first parameter is the image we want to analyze #the second one there is the action
alter = analyze['age']
emotion = analyze['dominant_emotion']
print(type(alter))
print(type(emotion))
print(analyze['race'])
print(alter)
print(emotion)