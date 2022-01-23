import cv2
from deepface import DeepFace
from pathlib import Path

# provide local path to the test-dataset
pathlist = Path(r'./data/emotion/test/neutral').rglob('*.png')
# create list with path and annotation of images
pictures = []
for path in pathlist:
    path_in_str = str(path)
    # adjust labeled value below
    pictures.append([path_in_str, "neutral"])

# variables to count true/false predictions and unrecognized faces
correct = 0
false = 0
no_face = 0

# scan & analyze each picture in list and compare with annotated emotion
for i in pictures:
    image = cv2.imread(i[0])
    try:
        analyze = DeepFace.analyze(image, actions=['emotion'], enforce_detection=True)
        if analyze['dominant_emotion'] == i[1]:
            correct += 1
        else:
            false += 1
    except:
        no_face += 1

# print results
print('Correct:', correct)
print('False:', false)
print('No Face Found:', no_face)