import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#Input Image
images = glob.glob('./Chessboard/*.jpg')
# print(len(images))

# videos = glob.glob('./Chessboard/*.mp4')
inputt = './Chessboard/chess4.mp4'
count=0
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(inputt))
if not capture.isOpened:
    print('Unable to open: ' + args.inputt)
    exit(0)

while True:
    if count>50:
        break
    ret, frame = capture.read()
    if frame is None:
        break
    # fname =frame
# for fname in images:
    img = frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        count+=1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        #break
cv2.destroyAllWindows()
#Analysis
print(count)
# print(objpoints)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("mtx: \n",mtx)
# print("ret: ",ret)
# print("rvecs: ",rvecs)
# print("tvecs: ",tvecs)

#----------Results------
"""
chess.mp4 -> 68
[[824.05762458   0.         381.10745975]
 [  0.         839.01299642 134.22842609]
 [  0.           0.           1.        ]]

 ches2.mp4 -> 1
 [[692.47904804   0.         439.49980877]
 [  0.         802.40991748 204.76436888]
 [  0.           0.           1.        ]]

 chess3.,p4 -> 0

 chess4.mp4 -> 229/51
 [[860.00199971   0.          27.8037219 ]
 [  0.         777.62721079 193.25869631]
 [  0.           0.           1.        ]]
"""
