# **Finding Lane Lines on the Road (andichik)** 

## Project Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.
1. Grayscale
2. Gaussian Blur
3. Canny
4. Region of Interest Mask
5. Hough Lines

Admittedly, I originally thought creating a region of interest mask should be the first step just because of its simplicity and immediate effect, but I quickly realized that adding a mask before applying canny transform would create a gradient line right along the region. Therefore, Grayscale->Gaussian Blur->Canny Transform were chose as my first three steps in the pipeline, and the region of interest mask as my fourth. Hough transform makes sense as the next, and last, step; the image has been processed to detect edges/borders that really matters and using Hough transform to extract/identify lines is the last piece of puzzle.


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by adding 2 steps of post-processing to average and extrapolate the lines identified by the pipeline. 

The first step was to go through all the lines identified by the pipeline and calculated weighted average lane line for each side. Since lines identified by the pipeline are passed as sets of pixel points, I calculated the slope and y-intercept and weighed these values based on the length, so that longer lines have larger effect on the averaged out lane lines.

The next step was to extend these weighted average lane lines to the edges of the region of interest. The previous step calculated the position of the line, so figuring out where the line would intersect with my region of interest was simple algebra. This step also involved converting float64 to int so that openCV can draw red lines in the correct pixels.

### 2. Identify potential shortcomings with your current pipeline

One shortcoming would be what would happen when the lane lines start to curve. Since the purpose of the altered draw_line() is to draw a single straight line for each side of lane lines, the lines would move around rapidly when the Hough transform identifies more curved lines along the path.

Another shortcoming would be the presence of guard rails or walls along the road that could be mistaken as lane lines. Canny and Hough transforms would almost definitely identify the edges of the walls as lines, and the region of interest mask may not be able to shave off irrelevant areas.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to add some sort of filter that would make the left and right lane line overlays less jittery due to occasional deviation. This could simply be a low pass filter, an extra step to normalize values, or a method to keep track of lane lines from previous frames to avoid sudden changes.

Another possible improvement would be to possibly quadratic regression to fit overlays right along a curved lane line.
