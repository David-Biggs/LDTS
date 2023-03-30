# LDTS

Local Density Threshold Shifting (LDTS) 

To obtain a count for objects in video or single frame, call video_count() and the count will be returned. 

For example:
  If we would like to count the detected objects in the array `my_detection_arrays`, which contains arrays for detections corresponding to frames in a video,     and we only want to use 50 frames from the video and a counting threshold of 95%, the function call could be as follows:
  
  `count = video_count(my_detection_arrays, 50, 0.95)'
