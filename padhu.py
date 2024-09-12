import cv2

img = cv2.imread("sai/img.png")
print(img.shape)

cv2.imshow("img",img)

cv2.waitKey(0)