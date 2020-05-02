from tensorflow.keras.models import load_model
import cv2
import face_recognition as fr
import numpy as np
import sys

model = load_model('trained/model.h5')

cap = cv2.VideoCapture(sys.argv[1])

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('result.avi', fourcc, 25, (640, 480))

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    locations = fr.face_locations(gray, model='cnn')

    faces = []

    for location in locations:
        y1, x2, y2, x1 = location

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        face = gray[y1:y2, x1:x2]
        face = cv2.resize(face, (26, 26))
        face = np.array(face) / 255
        face = np.reshape(face, (26, 26, 1))

        faces.append(face)

    faces = np.array(faces)

    if len(faces):
        results = model.predict(faces)

        for index, location in enumerate(locations):
            y, _, _, x = location
            result = results[index]
            result = np.argmax(result, -1)

            if result == 0:
                text = 'GOOD'
                color = (0, 255, 0)
            else:
                text = 'BAD'
                color = (0, 0, 255)

            cv2.putText(frame, text, (x, y), cv2.FONT_ITALIC, 1, color, 2)

    out.write(frame)

cap.release()
out.release()
