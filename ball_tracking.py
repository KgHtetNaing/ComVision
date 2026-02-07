import cv2 as cv
import time

print(cv.__version__)

def process_frame(frame):
    #BGR->HSV , Blur, inRange (to get color) , more clean up (open or close), find contour, filter out small contours , draw the contour
    scale_factor = 0.5
    small = cv.resize(frame , (int(frame.shape[1]*scale_factor) , int(frame.shape[0]*scale_factor)))
    hsv_frame = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    blur = cv.GaussianBlur(hsv_frame,(11,11),0)
    mask = cv.inRange(blur, (29,86 , 6) ,  (64, 255, 255))
    mask = cv.erode(mask, None, 2)
    mask = cv.dilate(mask, None , 2)
    cv.imshow("mask" , mask)

    contours,_ =  cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours , key = cv.contourArea)
        cv.drawContours(small, c , -1 , (0,255,0) , 3)
       
        M = cv.moments(c) #finding the center of the mass so we can put things on 
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"]))

            cv.circle(small, center, 10, (0,0,255), -1)
            cv.imshow("contours" , small)



#---------------------------------------------------------------
video = cv.VideoCapture(0)
time.sleep(1)

while True:
    
    ret , frame = video.read()

    if frame is None:
        break


    #processing
    #first step is to sperate foreground from background
    process_frame(frame)
    cv.imshow("frame",frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()