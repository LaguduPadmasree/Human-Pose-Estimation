import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture('vid/2.mp4')







class FaceDetector:
    def __init__(self, minDetectionCon=0.75):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.facedetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facedetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                imgheight, imgwidth,_ = img.shape
                bbox = int(bboxC.xmin * imgwidth), int(bboxC.ymin * imgheight), \
                       int(bboxC.width * imgwidth), int(bboxC.height * imgheight)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
                cv2.imshow('img', img)
        #return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5,rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255, 0, 255),rt)
        #top le
        cv2.line(img, (x, y), (x + l,y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        #top right
        cv2.line(img, (x1, y), (x1- l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        #bottom left
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        #bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img

    def main(self):
        cap = cv2.VideoCapture(0)
        pTime = 0
        detector = FaceDetector()
        while True:
            success, img = cap.read()
            detector.findFaces(img)
         #   #print(bboxs)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            #cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
            #cv2.imshow('img', img)
            cv2.waitKey(1)


