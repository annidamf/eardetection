import cv2
import dlib
from scipy.spatial import distance
import datetime

#function to calculate EAR based on formula
def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	eye_aspect_ratio = (A+B)/(2.0*C)
	return eye_aspect_ratio

videoFile = "videos_i8/10-1.mp4"
cap = cv2.VideoCapture(videoFile)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
total_frame = fps*duration
print(frame_count)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frame_no=1

while True:
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

	# Get date and time and
	# save it inside a variable
	dt = str(datetime.datetime.now())

	# put the dt variable over the
	# video frame
	frame = cv2.putText(frame, dt,
						(10, 100),
						font, 1,
						(210, 155, 155),
						4, cv2.LINE_8)

	faces = hog_face_detector(gray)
	#processing the landmark to detect left and right eyes
	for face in faces:
		face_landmarks = dlib_facelandmark(gray, face)
		leftEye = []
		rightEye = []

		#left eye coordinates (point 36 to 42)
		for n in range(36,42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			leftEye.append((x,y))
			next_point = n+1
			if n == 41:
				next_point = 36
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

		#right eye coordinates (point 42 to 48)
		for n in range(42,48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			rightEye.append((x,y))
			next_point = n+1
			if n == 47:
				next_point = 42
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

		left_ear = calculate_EAR(leftEye)
		right_ear = calculate_EAR(rightEye)

		EAR = (left_ear+right_ear)/2

		if (EAR<0.3):
			status = 0
		else:
			status=1

		to_print = ["frame : " +str(frame_no)+ " || timestamp: ", str(round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 2)) + " || EAR: ", str(EAR), " || status: ", str(status)]
		#EAR = round(EAR,2)
		with open('recordear101.txt', 'a+') as f:
			f.seek(0)
			data = f.read(100)
			if len(data)>0:
				f.write('\n')
			f.writelines(to_print)
			#f.write('{}'.format(EAR))
		#print(EAR)

	cv2.imshow("ear", frame)
	frame_no+=1

	key = cv2.waitKey(1)
	#if key == 27:
	if (key ==27) or (frame_no == frame_count):
		f.close()
		break

cap.release()
cv2.destroyAllWindows()
