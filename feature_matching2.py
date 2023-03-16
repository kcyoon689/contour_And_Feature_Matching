import cv2
import numpy as np

# queryImage
qimg = cv2.imread('./test2/F7_binary.png',0) 
# trainImage
timg = cv2.imread('./test2/F11_binary.png',0)
res1 = None
res2 = None

# SIFT 
orb = cv2.ORB_create()

kp1,des1 = orb.detectAndCompute(qimg,None)
kp2,des2 = orb.detectAndCompute(timg,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

res = cv2.drawMatches(qimg,kp1,timg,kp2,matches[:10],res1,flags=1000)

cv2.imshow("Feature Matching",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
