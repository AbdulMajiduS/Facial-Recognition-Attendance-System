import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime

print("⌛ Loading Encoded File...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnown, classNames = pickle.load(file)
print("✅ Encodings Loaded Successfully!")

cap = cv2.VideoCapture(0)

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        dataList = f.readlines()
        nameList = [line.split(',')[0] for line in dataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'{name},{dtString}\n')

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Facial Recognition Attendance', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
