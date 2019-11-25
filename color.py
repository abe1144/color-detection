from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import cv2
import pickle as pk


with open('./Data/lab_colors.pk', 'rb') as file:
    lab_colors = pk.load(file)

color_data = pd.read_csv('./Data/color_data.csv')


#function that takes in an image and reformates it for classification
def img_format(img):
    #read in image
    img = cv2.imread(img)
    #convert into RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #reshape image for less input
    img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    #Reshape image
    img = img.reshape(img.shape[0]*img.shape[1], 3)
    return img
    
#function that takes in image arr of rgb and outputs cluster centroids
def cluster(img_arr):
    db = DBSCAN(eps=10, min_samples=0.1*img_arr.shape[0]).fit(img_arr)
    labels = db.labels_
    #detects the number of clusters needed
    db_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    clt = KMeans(n_clusters = db_n_clusters)
    clt.fit(img_arr)
    #extract the counts for each cluster
    #labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    #counts, _ = np.histogram(clt.labels_, bins = labels)
    #RGB values of the cluster centers
    color_centroid = clt.cluster_centers_
    #list of rgb

    color_lst_rgb = list(color_centroid)
    return color_lst_rgb


#Function that returns closest colors to color clusters detected
def classify(rgb_arr):
    distance_arr = np.linalg.norm(color_data[['R','G','B']].sub(rgb_arr), axis=1)
    #get the index of the minimum value within the distance array
    min_idx = np.argmin(distance_arr)
    
    #using the index, return the actual color
    color = lab_colors[min_idx]
    return color


def classify_img(img):
    img_arr = img_format(img)
    centroid_lst = cluster(img_arr)
    color_lst = [classify(arr) for arr in centroid_lst]
    return color_lst