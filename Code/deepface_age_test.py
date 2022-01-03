import cv2
from deepface import DeepFace
from pathlib import Path

pathlist = Path('.\\data\\age\\25-30').rglob('*.jpg')
pictures = []
for path in pathlist:
    path_in_str = str(path)
    pictures.append([path_in_str, 50])

correct = 0
false = 0
no_face = 0

for i in pictures:
    image = cv2.imread(i[0])

    try:
        analyze = DeepFace.analyze(image, actions=['age'], enforce_detection=True)
        print(analyze['age'])
        if analyze['age'] > 19 and analyze['age'] < 36:
            correct += 1
        else:
            false += 1
    except:
        no_face += 1


print("Correct:", correct)
print("False:", false)
print("No Face Found:", no_face)