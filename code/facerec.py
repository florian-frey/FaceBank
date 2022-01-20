import cv2
from deepface import DeepFace
from deepface.extendedmodels import Age

imgpath = r"./data/pictures/seb.jpg"
image = cv2.imread(imgpath)

analyze = DeepFace.analyze(image,actions=['emotion','age', "gender"])
age = analyze['age']
emotion = analyze['dominant_emotion']
gender = analyze["gender"]

print(age, emotion, gender)