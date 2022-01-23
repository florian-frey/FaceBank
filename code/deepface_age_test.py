import cv2
from deepface import DeepFace
from pathlib import Path

# provide local path to the test-dataset
pathlist = Path(r'./data/age/test/22').rglob('*.png')
# create list with path and annotation of images
pictures = []
for path in pathlist:
    path_in_str = str(path)
    # adjust labeled age below
    pictures.append([path_in_str, 22])

# variables to count true/false predictions and unrecognized faces
correct = 0
false = 0
no_face = 0

# scan & analyze each picture in list and compare with real age
for i in pictures:
    image = cv2.imread(i[0])
    try:
        analyze = DeepFace.analyze(image, actions=['age'], enforce_detection=True)
        # allowing difference of 5 years
        if (i[1]-5) <= analyze['age'] <= (i[1]+5):
            correct += 1
        else:
            false += 1
    except:
        no_face += 1

# print results
print('Correct:', correct)
print('False:', false)
print('No Face Found:', no_face)