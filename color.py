from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

os.chdir("c:/users/abraham lin/desktop/")

img = cv2.imread("simple.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



#Optional Resize
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

plt.imshow(img)
#Reshape image
img = img.reshape(img.shape[0]*img.shape[1], 3)


clt = KMeans(n_clusters = 3)
clt.fit(img)

#extract the counts for each cluster
labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
counts, _ = np.histogram(clt.labels_, bins = labels)

#RGB values of the cluster centers
clt.cluster_centers_

#Extract most dominant color of image from # of counts

#detecting the closest color to commonly known colors
import csv
from colormath.color_objects import LabColor
reader = csv.DictReader('lab_matrix.csv')
lab_matrix = np.array([row.values() for row in reader])

# the reference color
color = LabColor(lab_l=69.34,lab_a=-0.88,lab_b=-52.57)

# find the closest match to `color` in `lab_matrix`
delta = color.delta_e_matrix(lab_matrix)
nearest_color = lab_matrix[np.argmin(delta)]
