# Python Color Detection using K-Means Clustering

Take a look at the Jupyter Notebook to see how it works. You can also check my <a href="https://deegoanalytics.herokuapp.com/blog/Clustering-Application--Color-Detection-in-Images"> blog post </a> on full details of how it works.



## Color.py Code

You can also import the ColorDetector Class under the `color.py` file. Here is how it works:

```
#initiate the class
colorDetector = ColorDetector()

#identify color in image

colorDetector.classify_img('./images/test_img.jpg')

#['white', 'black']

```