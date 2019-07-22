import cv2
import sys

Frame1 = None
video = cv2.VideoCapture(0)
check, Frame1 = video.read()
check, Frame = video.read()


while True:

	gray1 = cv2.cvtColor(Frame1,cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)
	GRAY1 = cv2.GaussianBlur(gray1,(21,21),0)
	GRAY2 = cv2.GaussianBlur(gray2,(21,21),0)


	delta_frame = cv2.absdiff(gray1,gray2)
	thresh_delta = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
	thresh_delta = cv2.dilate(thresh_delta,None,iterations=0)
	(cnts,_) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		if cv2.contourArea(contour) < 1000:
			continue
		(x,y,z,h) = cv2.boundingRect(contour)
		cv2.drawContours(Frame,cnts,-1,(0,255,0),2)
		#cv2.rectangle(Frame,(x,y),(x+z,y+h),(0,255,0),3)

	cv2.imshow("frame",Frame)
	cv2.imshow("capturing",GRAY2)
	cv2.imshow("delta",delta_frame)
	cv2.imshow("thresh",thresh_delta)
	
	Frame1 = Frame.copy()

	check, Frame = video.read()

	k = cv2.waitKey(1)
	if k == ord("q"):
		break
video.release()
cv2.destroyAllWindows()
sys.exit(0)
