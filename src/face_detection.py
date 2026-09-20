import cv2
camera = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
while True:
    success,frame=camera.read()
    if not success:
        print("Could not access camera")
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    for x,y,width,height in faces:

        face = frame[y:y+height,x:x+width]
        gray_face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        small_face = cv2.resize(gray_face, (48,48))
        cv2.imshow("Grayscale Face",gray_face)
        cv2.imshow("48x48 Face",small_face)
        cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)

    cv2.imshow("Face Detection",frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()