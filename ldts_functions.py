
import numpy as np
import os
import statistics as stats
from statsmodels.nonparametric.kernel_density import KDEMultivariate

def normalise(Z,upper_bound,lower_bound):
    '''
    Args:
        Z::[array_like]
            The array of value to normalise
        upper_bound:[float]
            Upper normalisation bound
        lower_bound:[float]
            Lower normalisation bound

    Return:
        Z_norm::[array_like]
            The array of the normalised values of Z within the lower_bound and upper_bound
    '''
    Z_norm = (upper_bound-lower_bound)*(np.array(Z) - np.min(np.array(Z)))/(np.max(np.array(Z))-np.min(np.array(Z))) + lower_bound

    return Z_norm


def LDTS(detection_data,threshold, upper_bound=1.00,lower_bound=0.00): 
    '''
    Args:
        detection_data::[array_like]
            n x 5 array with X1,Y1,X2,Y2,P as the coloumns. P is the classification probability. n is the number of detections 
        threshold::[float]
            The counting threshold
        upper_bound::[float]
            Upper normalisation bound. Default = 1
        lower_bound::[float]
            Lower normalisation bound. Default = 0 
    Returns:
        Count::[int]
            The number of counted objects within detection_data
    '''
    # Filter out P < 0.2
    cut_off_threshold = 0.20
    detection_data = detection_data[detection_data[:,4 ] > cut_off_threshold]
    # Get center points of bounding box
   
    X1 = detection_data[:,0]
    Y1 = detection_data[:,1]
    X2 = detection_data[:,2]
    Y2 = detection_data[:,3]
    P  = detection_data[:,4]
    
    # Get center of each bb
    X_center = np.add(X1,X2)/2
    Y_center = np.add(Y1,Y2)/2

    # Get density distribution f() and evaluate at the model f(X,Y)
    model = KDEMultivariate([X_center,Y_center],'cc',bw ="normal_reference")
    Z = model.pdf([X_center ,Y_center])
    Z_norm  = normalise(Z,upper_bound,lower_bound)
  
    # Shift the counting threshold
    Ts = threshold - Z_norm
   
    # Get the count
    # Count all detections with a classification prob greater than the shifted counting threshold
    count = np.count_nonzero(P > Ts)
    return count

def video_count(detection_data_path,max_frames,upper,lower,threshold,bw_method):
    '''
    Uses frames in a video to calculate average video count

    Args:
        detection_data_path::[str]
            Path to np arrays containing detection data - these should be the arrays from each frame in a video
        max_frames::[int]
            The number of frames to use when calculating the average 
        upper::[float]
            Upper normalisation bound. Default = 1.00
        lower::[float]
            lower normalisation bound. Default = 0.00
        threshold::[float]
            The counting threshold
        bw_method::[str]
            The bandwidth calculation method. Default 'normal_reference')
    Returns:
        average_count::[int]
            The average object count in the video

    '''
    
    frame_counter = 0
    video_counts = []
    norm_upper = upper
    norm_lower = lower
 
    for detection_arr in os.listdir(detection_data_path):

        if frame_counter < max_frames:          
            count  = LDTS(detection_arr,threshold,norm_upper, norm_lower,bw_method)
            video_counts.append(count)
            frame_counter +=1
        else:
            # If all frames have been used or if exceeded max_frames then break 
            break
            
    average_count = stats.mean(video_counts)
    # Return the average count for the video
    return average_count