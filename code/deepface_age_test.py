import cv2
from deepface import DeepFace
from pathlib import Path

pathlist = Path('.\\data\\pictures').rglob('*.jpg')
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
        analyze = DeepFace.analyze(image, actions=['age', "emotion"], enforce_detection=True)
        print(i[0],":", analyze['age'], analyze["dominant_emotion"])
        if analyze['age'] >= 55 and analyze['age'] <= 103:
            correct += 1
        else:
            false += 1
    except:
        no_face += 1


print("Correct:", correct)
print("False:", false)
print("No Face Found:", no_face)
