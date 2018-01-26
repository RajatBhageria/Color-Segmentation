import cv2, os
from barrelClassification import barrelClassification
folder = '2018proj1_test/'

for filename in os.listdir(folder):
    #read one test image
    img = cv2.imread(os.path.join(folder,filename))
    #apply barrelClassiciation
    x,y,d = barrelClassification(img)
    #print the results
    print "ImageNo = [%s], CentroidX = %d, CentroidY = %d, Distance = %.4f" % (filename, x,y,d)
