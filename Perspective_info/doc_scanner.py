import cv2 as cv
from imutils.perspective import four_point_transform
print (cv.__version__)

img = cv.imread("page.jpg")

height = 500
width = 500

small = cv.resize(img , (0 , 0), fx = 0.2 ,  fy = 0.2)  #fx and fy for scaling and containing quality
gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray , (5,5) , 0)
edged = cv.Canny (blurred , 75, 200)

cnts, _ = cv.findContours(edged , cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv.contourArea, reverse=True) [:5]
print (cv.contourArea(cnts[0]))

#counting corners of the object to determine the shape. 4 point for rect and 3 point for tri
c = cnts[0]
peri = cv.arcLength(c, True)
approx = cv.approxPolyDP(c , 0.02 * peri, True)

if len(approx) == 4:
    screen_cnt = approx

    contourIMG = small.copy()
    cv.drawContours(contourIMG, cnts, -1 , (0,0,255) , 2)
    warped = four_point_transform(small, screen_cnt.reshape(4,2))   #recheck this , change the paper into full screen
    cv.imshow("warp image", warped)
    cv.waitKey(1)

print (len(cnts))

cv.imshow("small image" , small)
cv.imshow ("edged image" , edged)
cv.imshow("gray image" , gray)

cv.waitKey()
cv.destoryAllWindows()