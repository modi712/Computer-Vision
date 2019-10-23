# Single projection original code

import argparse
import cv2
import numpy as np
import math
import os
from objloader_simple import *

# PARAMETERS
THRESHOLD = 10	# min number of matches to be recognized
# 105 for mark4, 65 - mark2
# for sift: 10 for mark4
SIZE = 3		# size for the display obj
# 3 for rat,fox, 1 for wolf, 100 for Rixa
ranthresh = 5.0	#5.0
#SIFT
sig = 2
loweRatio = 0.55	# 0.55 criteria for selectionf of features
bestMatchNumber = 2	#no of matches for points 2 for lowe ratio test
#PATHS
ref ='reference/mark2.jpg'
mod ='models/rat.obj'

#This functions loads the target surface image,
def main():

    homo = None
    l= None

#    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    camera_parameters = np.array([[824.05762458, 0, 381.10745975],[0, 839.01299642, 134.22842609],[0, 0, 1]])
    # create ORB/SIFT keypoint detector
#    orb = cv2.ORB_create()
    sift = cv2.xfeatures2d.SIFT_create(sigma=sig)#<>sigma
    
    # create BFMatcher object based on hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, ref), 0)
    # Compute model keypoints and its descriptors
#    kp_model, des_model = orb.detectAndCompute(model, None)
    kp_model,des_model = sift.detectAndCompute(model,None)
    kp_modelKP = kp_model
    kp_model = np.float32([k.pt for k in kp_model])
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, mod ), swapyz=True)
    # init video capture
    cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print ("Unable to capture video")
            return
        # find and draw the keypoints of the frame
        #orb
#        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        #sift
        kp_frame,des_frame = sift.detectAndCompute(frame,None)
        kp_frameKP = kp_frame
        kp_frame = np.float32([k.pt for k in kp_frame])
        # match frame descriptors with model descriptors
        try:
#        	matches = bf.match(des_model, des_frame)
        	matches = matcher(kp_model,kp_frame,des_model,des_frame)
        except:
        	print("Too Dark")
        	cap.release()
        	return 0

        # sort them in the order of their distance
        # the lower the distance, the better the match
#        matches = sorted(matches, key=lambda x: x.distance)

        # compute Homography if enough matches are found
        if len(matches) > THRESHOLD:
            # differenciate between source points and destination points
            print( "Enough matches found - %d/%d" % (len(matches), THRESHOLD) )
            #orb
#            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#            sift
            src_pts = np.float32([kp_model[i] for (_, i) in matches])
            dst_pts = np.float32([kp_frame[i] for (i, _) in matches])
            # compute Homography
            homo, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ranthresh)
            
            # Draw a rectangle that marks the found model in the frame
            if args.rectangle:
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homo)
                # connect them with lines
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                
            # if a valid homography matrix was found render cube on model plane
            if (homo is not None )and (not args.model):
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    (projection,l) = projection_matrix(camera_parameters, homo)
                    # project cube or model
                    frame = render(frame, obj, projection, model, False)
                    #frame = render(frame, model, projection)
                except:
                    pass
                    
            # print pose of camera
            if args.pose:
            	print('Pose of camera')
            	print(l)        
            # draw first 10 matches.
#            if args.matches:
#                frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
#                frame = cv2.drawMatches(model, kp_modelKP, frame, kp_frameKP, matches[:10], 0, flags=2)

            # show result
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print( "Not enough matches found - %d/%d" % (len(matches), THRESHOLD) )
            # draw first 10 matches.
#            if args.matches:
#                frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
#                frame = cv2.drawMatches(model, kp_modelKP, frame, kp_frameKP, matches[:10], 0, flags=2)
            # show result
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0
#---END Main---#

def matcher(kp1,kp2,features1,features2):
	matcher = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matcher.knnMatch(features1, features2, bestMatchNumber)
	# keeping only good matches wrt to lowe ratio ## check
	matches=[]
	for m,n in rawMatches:
		if m.distance < n.distance*loweRatio:
			matches.append((m.trainIdx,n.queryIdx))
	return matches

#Render a loaded obj model into the current video frame
def render(img, obj, projection, model, color=False):

    vertices = obj.vertices
    scale_matrix = np.eye(3) * SIZE
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
    return (np.dot(camera_parameters, projection),projection)
#---projection END---#

#Helper function to convert hex strings to RGB
def hex_to_rgb(hex_color):
    
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


# Command line argument parsing
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-mo','--model', help = 'do not draw model on target surface on frame', action = 'store_true')
parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
#parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
parser.add_argument('-po','--pose', help = 'print camera pose for each frame', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
