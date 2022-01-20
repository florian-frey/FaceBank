import cv2
from deepface import DeepFace
from pathlib import Path

# provide local path to the test-dataset
pathlist = Path(r'./data/emotion/test/neutral').rglob('*.png')
pictures = []
for path in pathlist:
    path_in_str = str(path)
    # adjust labeled value
    pictures.append([path_in_str, "neutral"])

correct = 0
false = 0
no_face = 0

for i in pictures:
    image = cv2.imread(i[0])

    try:
        analyze = DeepFace.analyze(image, actions=['emotion'], enforce_detection=True)
        print(analyze["dominant_emotion"])
        if analyze['dominant_emotion'] == i[1]:
            correct += 1
        else:
            false += 1
    except:
        no_face += 1

    

print("Correct:", correct)
print("False:", false)
print("No Face Found:", no_face)