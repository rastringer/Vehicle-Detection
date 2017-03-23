# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Writeup 
[//]: # (Image References)
[image3]: ./output_images/initial_window_image0.jpg "Initial Windows"
[image4]: ./output_images/initial_window_image1.jpg "Initial Windows"
[image5]: ./output_images/initial_window_image2.jpg "Initial Windows"
[image6]: ./output_images/processed_image_2ver4.jpg "Overlap"
[image7]: ./output_images/processed_image_4ver4.jpg "Overlap"
[image8]: ./output_images/processed_image_4ver0.jpg "Pipeline"
[image9]: ./output_images/processed_image_4ver1.jpg "Pipeline"
[image10]: ./output_images/processed_image_4ver2.jpg "Pipeline"
[image11]: ./output_images/video1.PNG "Video Frame"
[image12]: ./output_images/video2.PNG "Video Frame"
[image13]: ./output_images/video3.PNG "Video Frame"

<!-- 
window_img function results

[image1]: ./output_images/initial_window_image0.jpg "Initial Window Applied"
[image1]: ./output_images/initial_window_image0.jpg "Initial Window Applied"
[image1]: ./output_images/initial_window_image0.jgp "Initial Window Applied"
 -->
### Histogram of Oriented Gradients (HOG)

I read in all vehicle and non-vehicle images, separating them into two classes. I found the best results using YCrCb color parameters and turning on "ALL" hog channels. I experimented with different channels however ALL seemed the failsafe option. I found training gains by augmenting the number of orients beyond 9. I set spatial and histogram bins to 32.

```
# Parameters - modify and check changes in results
# Windows
xy_overlap = (0.8, 0.8)
xy_window = [64, 64]
y_start_stop = [340, 680] # Min and max y for search in slide_window()
x_start_stop = [760, 1260]
# Color channels, hog and histograms
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
vis = True
```

I settled on these parameters as they provided far shaper images and vehicle detection than other combinations.

### Training a classifer using HOG Features

I trained a linear SVM using hog-features, spatial and color-histogramm features. Initial accuracies ranged from 0.97-0.98, and passed 0.99 with some work on the parameters.


### Sliding window search 

Sliding window search required a fair amount of testing, checking images and experimenting with new parameters. 
Here are some inital images showing the windows, covering not only the cars but trees and road features:

![alt text][image3]
![alt text][image4]
![alt text][image5]

After altering parameters to have the algorithm ignore the sky here:

```
y_start_stop = [340, 680] # Min and max y for search in slide_window()
```

I turned my attention to the overlapping. Using a 0.5 overlap confused the program, and it began boxing spaces between cars:

![alt text][image6]

An overlap of 0.8 improved the situation, and tuning the overlap, y_start_stop and window size vastly helped decrease false positives:

![alt text][image7]

### Images and Optimization

Working with the YCrCb color channel, with "ALL" HOG features, spatially binned color and color histograms in the features vector gave the best results.

![alt text][image8]
![alt text][image9]
![alt text][image10]

### Video Implementation

Here's a [link to my video result](https://youtu.be/I2N8s3K1ktI)

### Filtering False Positives, Bounding Boxes

I created a heatmap from recordings of positive detection positions from each video frame. The heatmap is then thresholded to identify vehicle positions. I used scipy.ndimage.measurements.label() to label the individual groupings in the heatmap, which are likely to be vehicles. Bounding boxes coer the area of the detected groupings.
<!-- 
Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]
 -->
Here the resulting bounding boxes are drawn onto the last frame in the series and on other frames in the video:

![alt text][image11]
![alt text][image12]
![alt text][image13]

### Discussion

As with prior computer vision projects, I wonder if there can ever be a concrete pipeline to deal with every eventuality. There are so many variables in terrain, color, conditions and angles. I would like to replicate this project also using a deep learning model to compare results. 
Based on the approach taken here however, I would like to work on  averaging the boxes over several frames to make the results less jittery. Experimentation with narrowing the search of the image to the left side of the screen may also be in order for this video.  
