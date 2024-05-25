import cv2
import os
import numpy as np
import pandas as pd
import datetime
import time
from tensorflow import keras
from keras.models import load_model

model = load_model('trained_model.h5')
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
directory="D:/Downloads/project/dataset"

entries = os.listdir(directory)
roll_nums=[]

for entry in entries:
    if os.path.isdir(os.path.join(directory, entry)):
        roll_nums.append(entry)
print(roll_nums)

attendance_sheet = pd.read_excel('attendance.xlsx') if os.path.exists('attendance.xlsx') else pd.DataFrame(columns=['Name'])

todays_date = datetime.date.today().strftime('%Y-%m-%d')
capture = cv2.VideoCapture(0)

duration = 20

end = time.time() + duration

attendance = {label: 0 for label in roll_nums}
print(attendance)

while (time.time()) < end:
    ret, frame = capture.read()
    frame=cv2.flip(frame,1)
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x_cord, y_cord, width, height) in faces:
        captured_img = frame[y_cord:y_cord+height, x_cord:x_cord+width]
        
        captured_img = cv2.resize(captured_img, (224, 224))
        captured_img = captured_img / 255.0
        captured_img = np.expand_dims(captured_img, axis=0)
        
        probability = model.predict(captured_img)
        
        class_idx = np.argmax(probability)
        class_label=roll_nums[class_idx]
        attendance[class_label] += 1
        
        cv2.rectangle(frame, (x_cord, y_cord), (x_cord+width, y_cord+height), (0, 255, 0), 2)
        cv2.putText(frame, class_label, (x_cord, y_cord-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Attendance', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

max_count=max(attendance.values())
#print(attendance)

for label, count in attendance.items():
    if count >= (max_count*0.6):
        attendance_sheet.loc[attendance_sheet['Name'] == label, todays_date] = 'Present'
    else:
        attendance_sheet.loc[attendance_sheet['Name'] == label, todays_date] = 'Absent'

attendance_sheet.to_excel('attendance.xlsx', index=False)

print("Attendance updated for", todays_date)
print(attendance_sheet)
