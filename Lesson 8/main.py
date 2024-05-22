import cv2

path = "Lesson 8/datasets/ishwa"
faces = "Lesson 8/images/face.xml"
model = cv2.CascadeClassifier(faces)
webcam = cv2.VideoCapture(0)

print(webcam.read())
count = 0
while count < 30:
    ret,video = webcam.read()
    face = model.detectMultiScale(video, 1.5, 4)
    #print(face)
    for (x,y,w,h) in face:
        rectangle = cv2.rectangle(video,(x,y),(x+w, y+h),(0,0,0), 3)
        person = video[y:y+h, x:x+w]
        person_resize = cv2.resize(person,(130,100))
        cv2.imwrite(path+"/"+str(count)+".png", person_resize)
        count = count + 1

    cv2.imshow("Video",video)
    k = cv2.waitKey(10)
    if k == 27:
        break
