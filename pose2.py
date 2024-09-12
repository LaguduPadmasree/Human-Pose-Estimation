import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from cvzone.FaceMeshModule import FaceMeshDetector

detector = FaceMeshDetector(maxFaces=1)
def calculate_angle(a, b, c):
    a = np.array(a)  # First


    b = np.array(b)  # Mid
    c = np.array(c)  # En

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)
length2=0
length4=0
length6=0
height, width = 480, 640  # Specify the height and width of the window
#black_image = np.zeros((height, width, 3), dtype=np.uint8)  # Create a black image (3-channel RGB)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        try:
            ret, frame = cap.read() #read video
            black_image = np.zeros((height, width, 3), dtype=np.uint8) #create special window

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#convert color image into black and white

            image.flags.writeable = False

            # Make detection
            results = pose.process(image)# find the body in the image

            # Recolor back



            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark # we can get pose landmarks in this
               #print(landmarks)
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), #connecting lines between points and it will give the color and thicness

                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            '''for lndmrk in mp_pose.PoseLandmark:
                print(lndmrk)'''

            Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]    #extracting values from landmark like x,y for finding angle and distance

            Lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            Lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

            Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            LEFT_KNEE= [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            LEFT_ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            length, _ = detector.findDistance(Lwrist,Rwrist)
            length=length*100
            #print(length)

            length1, _ = detector.findDistance( LEFT_HIP, Lwrist)
            length1 = length1 * 100

            length1, _ = detector.findDistance(LEFT_HIP, Lwrist)
            length1 = length1 * 100

            #print(length1,length2)
            if length1<10:
                i="Your left hand is downside"
            elif length1>length2:
                i="Your left hand is going up"

            elif length1<length2:
                i='Your left and is going down'
            length2 = length1


            length3, _ = detector.findDistance(RIGHT_HIP, Rwrist)
            length3 = length3 * 100
            #print(length3)
            if length3<12:
                j="Your Right hand is downside"
                #print(j)
            elif length3>length4:
                j="Your Right hand is going up"

            elif length3<length4:
                j='Your Right Hand is going down'
            length4 = length3

            length5, _ = detector.findDistance(LEFT_HIP, LEFT_ANKLE)
            length5 = length5 * 100
            #print(length5)

            if length5>30:
                h="standing"
            elif length5<30:
                h="sitting down"

            cv2.putText(black_image, f'{i}', (20, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 255), 4)  #putting text in black window
            cv2.putText(black_image, f'{j}', (20, 400), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 255), 4)
            cv2.putText(black_image, f'{h}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 255), 4)



            #print(shoulder)
            Left_hand_angle=calculate_angle(Lshoulder, Lelbow, Lwrist)
            Right_hand_angle = calculate_angle(Rshoulder, Relbow, Rwrist)  #calculatting angle between 3 points
            LEFT_LEG = calculate_angle(LEFT_HIP,LEFT_KNEE,LEFT_ANKLE)
            LEFT_KNEE = calculate_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)





            #cv2.putText(image, f'distance between to hands:{int(length)}', (20,200), cv2.FONT_HERSHEY_PLAIN,
                        #2, (255, 0, 0), 2)


            if length>=70:
                hug="hug me"
                cv2.putText(black_image, f'{hug}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0,0),4)







            cv2.imshow('Mediapipe Feed', image)
            cv2.imshow('Mediapipe Feed1', black_image)
           # print('hi')

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        except:
            pass
