import cv2

framewidth = 840
frameheight = 730

cap = cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4,frameheight)

while 1:
    sucess,img = cap.read()
    img = cv2.resize(img,(framewidth,frameheight))
    print(img)
    cv2.imshow("viedo",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break