#Ping pong game

import argparse

import cv2
import numpy as np
import math
import os
from objloader_simple import *

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 40
ONE = True
TWO = True
referencePath ='reference/mark17.jpg'
referencePath2 = 'reference/mark18.jpg'
#image = cv2.imread('Images/3.jpg')

def main():
    """
    This functions loads the target surface image,
    """
    homography = None
    homography2 = None
    # matrix of camera parameters (made up but works quite well for me)
    # calculate camera_parameters???
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, referencePath), 0)
    model2 = cv2.imread(os.path.join(dir_name, referencePath2), 0)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    kp_model2, des_model2 = orb.detectAndCompute(model2, None)
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)
    obj2 = OBJ(os.path.join(dir_name, 'models/rat.obj'), swapyz=True)
    # init video capture
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #out = cv.VideoWriter('/home/nihar/Desktop/abc/middle.avi', -1, 20.0, (640,480))
    out = cv2.VideoWriter('./pingpong9s.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    t = 0
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    q = 0

    p1 = np.zeros((3,4))
    p2 = np.zeros((3,4))
    p = np.zeros((3,4))
    p2 = np.zeros((3,4))
    pl1 = np.zeros((3,4))
    pl2 = np.zeros((3,4))
#    frame = image   
#    # find and draw the keypoints of the frame
#    kp_frame, des_frame = orb.detectAndCompute(frame, None)
#    # match frame descriptors with model descriptors
#    try:
#    	matches = bf.match(des_model, des_frame)
#    	matches2 = bf.match(des_model2, des_frame)
#    #Fun way of closing the program
#    except:
#    	print("Closing")
##       	cap.release()
#       	return 0
#        	
        	     
    while True:
	
	if t>1.5:
	    break 

        # read the current frame
        ret, frame = cap.read()
        #image1 = cv2.imread('Images/1.jpg')
        #frame = image1
        if not ret:
            print ("Unable to capture video")
            return
        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        try:
        	matches = bf.match(des_model, des_frame)
        	matches2 = bf.match(des_model2, des_frame)
        #Fun way of closing the program
        except:
        	print("Closing")
        	cap.release()
        	return 0

        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)
        matches2 = sorted(matches2, key=lambda x: x.distance)
#------------
        # compute Homography if enough matches are found
        if len(matches) > MIN_MATCHES  and ONE:
            # differenciate between source points and destination points
            print( "Enough matches found - %d/%d" % (len(matches), MIN_MATCHES) )
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if args.rectangle:
                # Draw a rectangle that marks the found model in the frame
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography)
                # connect them with lines
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            # if a valid homography matrix was found render cube on model plane
            if (homography is not None )and (not args.model):
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection1, l1 = projection_matrix(camera_parameters, homography)
		    
		    if counter1 == 0:
			p1 = projection1
			pl1 = l1
			counter1 = 1
			
                    # project cube or model
                    #frame = render(frame, obj, projection1, model, False)
                    #frame = render(frame, model, projection)
                except:
                    pass
            # draw first 10 matches.
            #if args.matches:
            #    frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
            # show result
            #cv2.imshow('frame', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

        else:
            print( "Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES) )
#----END First Object
#----Second object------------2222222222222222222222-------------------
#------------


        # compute Homography if enough matches are found
        if len(matches2) > MIN_MATCHES and len(matches2) <= len(matches) and TWO :
            print( "Enough matches2 found - %d/%d" % (len(matches2), MIN_MATCHES) )
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model2[m.queryIdx].pt for m in matches2]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches2]).reshape(-1, 1, 2)
            # compute Homography
            homography2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if args.rectangle:
                # Draw a rectangle that marks the found model in the frame
                h, w = model2.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography2)
                # connect them with lines
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            # if a valid homography matrix was found render cube on model plane
            if (homography2 is not None )and (not args.model):
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    #homography2[:,-1] = t*(homography2[:,-1]-homography[:,-1]) + homography[:,-1]
                    #print(homography)
                    projection, l = projection_matrix(camera_parameters, homography2)
		    if counter2 == 0:
			p2 = projection
			pl2 = l
			counter2 = 1
#                    projection1[:,-1] = t*(projection[:,-1]-projection1[:,-1]) + projection1[:,-1]

		    #l2 = pl1-pl2
		    #pl1 = pl1
		    #pl1[1,3] = pl1[1,3] - 2*pl2[1,3]
		    #m = t
		    #if 2>t>1:
		    #	p1 = p2
		    #	p2 = np.dot(camera_parameters, pl1)
	 	    #	m = t-1


		    
		   
                    # project cube or model
                    #frame = render(frame, obj2, projection, model2, False)
                    #frame = render(frame, model, projection)
		    if counter4 == 1:
			l2 = pl2-pl1
		        pl1 = pl1
		        pl2[1,3] = pl2[1,3] - 2*l2[1,3]
		        m = t
		    
		    	p2 = p1
		    	p1 = np.dot(camera_parameters, pl2)
	 	    
			#p = p2
			#p2 = p1
			#p1 = p
			#projection1 = projection
			counter4 = 0
                    #projection3 = t*(p1-p2) + p2


		    if counter4 == 2:
			l2 = pl1-pl2
		        pl1 = pl1
		        pl1[1,3] = pl1[1,3] - 2*l2[1,3]
		        m = t
		    
		    	p2 = p2
		    	p1 = np.dot(camera_parameters, pl1)
			#p = p2
			#p2 = p1
			#p1 = p
			#projection1 = projection
			counter4 = 0
                    
		    projection3 = t*(p1-p2) + p2


		    #print(np.linalg.norm(projection3 - projection))
		    if(counter3 == 1):
			print(np.linalg.norm(projection3 - projection1))
		    	if np.linalg.norm(projection3 - projection) < 60000:
		    
				counter1 = 0
				print('hello')
				counter2 = 0
				counter3 = 0
				counter4 = 2
				t = 0
				q += 0.01
		    if(counter3 == 0):
			print(np.linalg.norm(projection3 - projection1))
		    	if np.linalg.norm(projection3 - projection1) < 40000:
				print('hi')
		    
				counter1 = 0
				counter2 = 0
				counter3 = 1
				counter4 = 1
				t = 0
				q += 0.01
                    t += q + 0.01
                    # project cube or model
                    frame = render(frame, obj2, projection3, model2, False)
                    #frame = render(frame, model, projection)
                except:
                    pass
            # draw first 10 matches.
            if args.matches:
                frame = cv2.drawMatches(model2, kp_model2, frame, kp_frame, matches2[:10], 0, flags=2)
            # show result
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print( "Not enough matches2 found - %d/%d" % (len(matches2), MIN_MATCHES) )

#--END Second---------#
#    cap.release()
    cv2.destroyAllWindows()
    return 0
    
#---------------END Main()------------------#

#Render

def render(img, obj, projection, model, color=True):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.5
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return (np.dot(camera_parameters, projection), projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-mo','--model', help = 'do not draw model on target surface on frame', action = 'store_true')
parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
# TODO jgallostraa -> add support for model specification
#parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
