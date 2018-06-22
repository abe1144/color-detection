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
x = clt.cluster_centers_

#Extract most dominant color of image from # of counts


#######Color classifier

def classify(rgb):
    
    colors = {"red": (255, 0, 0),
              "green": (0,255,0),
              "blue": (0,0,255),
              "pink": (255,20,147),
              "purple": (128,0,128),
              "gray": (128,128,128),
              "black": (0,0,0),
              "white": (255,255,255),
              "teal": (0,128,128),
              "orange": (255,165,0)}
    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) 
    distances = {k: manhattan(v, rgb) for k, v in colors.items()}
    color = min(distances, key=distances.get)
    return color