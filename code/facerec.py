import cv2
from deepface import DeepFace

# define path to image and read image
imgpath = r'./data/seb.jpg'
image = cv2.imread(imgpath)

# analyze image, save attributes and print them
analyze = DeepFace.analyze(image, actions=['age', 'emotion', 'gender', 'race'])

age = analyze['age']
emotion = analyze['dominant_emotion']
gender = analyze['gender']
race = analyze['dominant_race']

print(age, emotion, gender, race)