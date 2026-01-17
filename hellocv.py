import cv2

image = cv2.imread("C:/Users/Student/Documents/Comvision/assets/squirtle.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#HSV
#LAB

print(image)
cv2.imshow("squirtle" , image) 
cv2.imshow("greyimg" , gray_image)
cv2.imshow("hsvimg" , hsv_image)
cv2.imshow("lab_image" , lab_image)
cv2.waitKey(0) #wait for image to show