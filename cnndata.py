import cv2
import os

directory = 'D:/Downloads/project/dataset'

student_name = input("Enter the student's name: ")

student_directory = os.path.join(directory, student_name)
if not os.path.exists(student_directory):
    os.makedirs(student_directory)

cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

count = 0
while count < 450:
    ret, frame = capture.read()
    frame=cv2.flip(frame,1)
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x_cord, y_cord, width, height) in faces:
        cv2.rectangle(frame, (x_cord, y_cord), (x_cord+width, y_cord+height), (0, 255, 0), 2)

        face_img = frame[y_cord:y_cord+height, x_cord:x_cord+width]
        filename = os.path.join(student_directory, f'{student_name}_{count}.jpg')
        cv2.imwrite(filename, face_img)
        count += 1
    
    cv2.imshow('capture Images', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
