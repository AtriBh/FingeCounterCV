import os
import cv2
import time
import HandTrackingModule as htm

wCam, hCam = 640,480

cap = cv2.VideoCapture(0)

cap.set(3,wCam)
cap.set(4,hCam)

folderPath = "fingerimages-copy"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(cv2.resize(image,(200,200)))

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]
totalFingers = 0

while True:
    sucess,img = cap.read()

    img = detector.findHands(img)

    lmList = detector.findPosition(img,draw = False)

    print(lmList)

    if len(lmList) !=0 :

        fingers = []

        # Thumb
        #id no 1 is the x axis while id no 2 is the y axis
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:  # Index finger tip above joint
            fingers.append(1)
        else:
            fingers.append(0)
        
        # checking when the 4 fingers are closed or open

        for id in range(1,5):
            # id and id -2
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:  # Index finger tip above joint
                fingers.append(1)
            else:
                fingers.append(0)
        
        # print(fingers)

        # find how many 1s are there
        totalFingers = fingers.count(1)

        #print(totalFingers)
        
                

    # palcing the image within our original image
    h,w,c = overlayList[totalFingers].shape
    img[0:h,0:w] = overlayList[totalFingers]

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,f'FPS {int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image",img)
    
    

    # 1 ms delay
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

