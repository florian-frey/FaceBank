import cv2
from deepface import DeepFace
from pathlib import Path

# provide local path to the test-dataset
pathlist = Path(r'./data/age/22').rglob('*.png')
pictures = []
for path in pathlist:
    path_in_str = str(path)
    # adjust labeled age
    pictures.append([path_in_str, 22])

correct = 0
false = 0
no_face = 0

for i in pictures:
    image = cv2.imread(i[0])

    try:
        analyze = DeepFace.analyze(image, actions=['age'], enforce_detection=True)
        if (i[1]-5) <= analyze['age'] <= (i[1]+5):
            correct += 1
        else:
            false += 1
    except:
        no_face += 1


print("Correct:", correct)
print("False:", false)
print("No Face Found:", no_face)