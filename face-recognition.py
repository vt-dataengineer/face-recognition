import cv2
from random import randrange

# load pre trained data from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces
# img = cv2.imread('cr7.jpg')

# capture video from webcam
webcam = cv2.VideoCapture(0)

# loop over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectanglearound faces
    for (x,y,w,h) in face_coordinates:
      cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)), 2)


    cv2.imshow('Face recognition', frame)
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key==81 or key==113:
        break

# release the webcam
webcam.release()

print('Code Completed')    
