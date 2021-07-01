"""
@author: Jimmy Ardoin
"""

import os
import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

def tiff_getter():
    """
    Gets the values of all .tiff files in the directory and subdirectories
    according to the original specificaitons of the files.
    
    Returns
    -------
    exposure_dict : dict
        The exposure times paired with all the light frames matching it.
    dark_dict : dict
        The exposure times paired wwith all the dark frames matching it.
        
    """
    exposure_dict = {}
    dark_dict = {}
    count = 0
    path = input('Please enter the exact path to the directory containing all the .tiff files')
    
    for root, dirs, files in os.walk(path):
        image_list = []
        image_arr_list = []
        for file in files:
            if file.endswith('.tiff'):
                image_list.append(str(root) + '/' + str(file))
                
        image_arr_list = [np.array(Image.open(file)).astype(np.int32) \
                           for file in image_list]
        (Image.close(file) for file in image_list)
        
        if root[-3] == 'p' and '\\dark\\' in root:
            dark_dict['0.' + root[-2]] = [image for image in image_arr_list]
        elif root[-3] =='p':
            exposure_dict['0.' + root[-2]] = [image for image in image_arr_list]
        elif '\\dark\\' in root and root[-4:-1] != '\\dark\\':
            dark_dict[root[-2]] = [image for image in image_arr_list]
        elif count != 0  and '\\dark\\' not in root and '\\dark' not in root:
            exposure_dict[root[-2]] = [image for image in image_arr_list]
        count += 1
        
    del exposure_dict['0']
    del dark_dict['0']
    return (exposure_dict, dark_dict)

def mean_iterator(dictionary):
    """
    Iterates through the dictionary of sub-stacks to give back a dictionary of mean images.
    Then filters through images with sigma clpping to reject outliers.
    
    Parameters
    ----------
    dictionary : dict
        A dictionary of lists of arrays with whatever key you like

    Returns
    -------
    mean_dict : dict
        A dictionary of the mean images of dark_dictionary sub_arrays with the
        exposures as keys.

    """
    
    print('Creating mean images')
    
    #Iterating through dictionary for mean
    mean_dict = {}
    mean_dict_unfil={}
    for exp, sub_arr in dictionary.items():
        sub_stack = np.stack(sub_arr)
        mean_dict_unfil[exp] = np.mean(sub_stack, axis=0)
        
    #Filtering results with sigma clipping
    print('Sigma clipping mean images')
    clip_criterion = 5
    for exp, sub_arr in mean_dict_unfil.items():
        median = np.median(sub_arr)
        stdev = np.std(sub_arr)
        sub_arr = np.where(sub_arr > (median + stdev*clip_criterion), median, sub_arr)
        sub_arr = np.where(sub_arr < (median - stdev*clip_criterion), median, sub_arr)
        mean_dict[exp] = sub_arr
        
    return(mean_dict)

def regressor(x_list, y_list):
    """
    Takes lists, one of which must be a list of np.ndarrays, and regresses them
    pixel by pixel. The arrays can be either or both inputs. Returns the images
    of the regression.
    
    Parameters
    ----------
    x_list, y_list : list
        The input lists, one or both of which must be a list of arrays
    
    Returns
    ----------
    m_arr : ndarray
        The array image of slopes for each pixel.
    b_arr : ndarray
        The array image of y-intercepts for each pixel
    """
    
    #Bool initialization
    x_check = False
    y_check = False
    
    #Checking which inputs are np.ndarrays
    if isinstance(x_list[0], np.ndarray):
        x_stack = np.stack(x_list)
        x_check = True
        
    if isinstance(y_list[0], np.ndarray):
        y_stack = np.stack(y_list)
        y_check = True
    
    #Going through each case of the above and generating images
    if x_check and y_check:
        
        index = np.zeros(y_list[0].shape)
        m_arr = np.empty(y_list[0].shape)
        b_arr = np.empty(y_list[0].shape)
        
        for i, j in np.argwhere(index==0):
            x_vals = x_stack[:,i,j]
            y_vals = y_stack[:,i,j]
            m_arr[i,j], b_arr[i,j] = np.polyfit(x_vals, y_vals, deg = 1)
        
        
    elif x_check:
        
        index = np.zeros(x_list[0].shape)
        m_arr = np.empty(x_list[0].shape)
        b_arr = np.empty(x_list[0].shape)
        
        for i, j in np.argwhere(index==0):
            x_vals = x_stack[:,i,j]
            m_arr[i,j], b_arr[i,j] = np.polyfit(x_vals, y_list, deg = 1)
            
            
    elif y_check:
        
        index = np.zeros(y_list[0].shape)
        m_arr = np.empty(y_list[0].shape)
        b_arr = np.empty(y_list[0].shape)
        
        for i, j in np.argwhere(index==0):
            y_vals = y_stack[:,i,j]
            m_arr[i,j], b_arr[i,j] = np.polyfit(x_list, y_vals, deg = 1)
            
            
    else:
        print('Oops! regressor() requires at least one input to be an np.ndarray')
        sys.exit()
        
    return(m_arr, b_arr)

def transfer_processor(exposure_dict):
    #List initialization
    rms_list = []
    mean_list = []
    var_list = []

    #Filling lists
    for arrs in exposure_dict.values():
        stack = np.stack(arrs)
        stack = np.squeeze(stack)
        rms_list.append(np.sqrt(np.mean(np.square(stack), axis=0)))
        mean_list.append(np.mean(stack, axis=0))
        var_list.append(np.var(stack, axis=0))
        
    #Performing 
    print('Regressing transfer curve data. This will take a while')
    gain, read_noise = regressor(mean_list, var_list)
    gain_shape = gain.shape
    gain = [1/m for m in gain.flatten()]
    gain = np.reshape(gain, gain_shape)
    
    #Plotting raw results
    print('Now plotting results')
    fig1 = plt.figure()
    plt.title('Gain Map')
    ax1 = fig1.add_subplot(111)
    cax = ax1.matshow(gain, cmap='plasma')
    fig1.colorbar(cax, label='Gain (electrons/ADU)')
    plt.text(10,80,'Average gain :' + str(np.mean(gain))[0:5])
    plt.show()
    plt.close()
    
    #Sigma clipping
    clip_criterion = 4
    
    median = np.median(gain)
    stdev = np.std(gain)
    gain = np.where(gain > median + stdev*clip_criterion, median, gain)
    gain = np.where(gain < median - stdev*clip_criterion, median, gain)
    
    median = np.median(read_noise)
    stdev = np.std(read_noise)
    read_noise = np.where(read_noise > median + stdev*clip_criterion, median, read_noise)
    read_noise = np.where(read_noise < median - stdev*clip_criterion, median, read_noise)
    
    #Replotting
    fig1 = plt.figure()
    plt.title('Gain Map with sigma clipping')
    ax1 = fig1.add_subplot(111)
    cax = ax1.matshow(gain, cmap='plasma')
    fig1.colorbar(cax, label='Gain (electrons/ADU)')
    plt.text(10,80,'Average gain :' + str(np.mean(gain))[0:5])
    plt.show()
    plt.close()

    fig2 = plt.figure()
    plt.title('Read Noise Map')
    ax2 = fig2.add_subplot(111)
    cax = ax2.matshow(read_noise, cmap='plasma')
    fig2.colorbar(cax, label='ADU')
    plt.show()
    plt.close()
    
    
    plt.figure()
    histo = gain.flatten()
    plt.title('Pixel gain histogram')
    plt.xlabel('Pixel gain')
    plt.ylabel('Instances')
    plt.hist(histo, bins='auto')
    plt.show()
    plt.close()
    return

def linearity_processor(mean_dict):
    
    print('Beginning linearity processing')
    exposure = [float(exp) for exp in list(mean_dict.keys())]
    
    print('Regressing linearity values. This will take a while')
    #This is if you want a regression of the whole iamge.
    m_arr, b_arr = regressor(exposure, list(mean_dict.values()))
 
    #Preparations for linearity curves
    indices = np.argwhere(np.zeros(list(mean_dict.values())[0].shape)==0)
    num_pixels = 35   #How many curves you want plotted
    np.random.shuffle(indices)
    indices_list = indices.tolist()[0:num_pixels]
    ratios = [[] for num in indices_list]
    mean_list = [[] for num in indices_list]
    mean_stack = np.stack(list(mean_dict.values()))
    
    #Iterating through the randomly sampled indices to get the ratios.
    count=0
    for index_1, index_2 in indices_list:
        measured_list = mean_stack[:, index_1, index_2]
        mean_list[count] = measured_list
        expected_list = [exp*m_arr[index_1, index_2] 
                           + b_arr[index_1, index_2] for exp in exposure]
        temp_ratios = [i/j for i, j in zip(measured_list, expected_list)]
        ratios[count] = temp_ratios
        count += 1
    
    plt.figure()
    exposure_vals = [exp*m_arr[12, 55] + b_arr[12,55] for exp in exposure]
    plt.plot(exposure, exposure_vals, color='blue')
    plt.scatter(exposure, mean_stack[:, 12, 55], c='red')
    plt.show()
    plt.close()
    
    #Plotting the curves
    plt.figure()
    for mean_sublist, ratio_sublist in zip(mean_list, ratios):
        plt.plot(mean_sublist, ratio_sublist)
        
    plt.title('Linearity Curves')
    plt.xlabel('Mean counts at different exposures')
    plt.ylabel('Measured/Expected Pixel value')
    plt.hlines(1, 0, np.max(mean_list[0]), linestyles='dashed')
    plt.ylim(0.7, 1.3)
    plt.show()
    plt.close()
    
    # plt.figure()
    # for exp, ratio_sublist in zip(exposure, ratios):
    #     plt.plot(exp, ratio_sublist)
        
    # plt.title('Linearity Curves')
    # plt.xlabel('Exposure time')
    # plt.ylabel('Measured/Expected Pixel value')
    # #plt.hlines(1, linestyles='dashed')
    # plt.show()
    # plt.close()
    
def Main():
    
    exposure_dict, dark_dict = tiff_getter()
    
    mean_dark_dict = mean_iterator(dark_dict)
            
    #Dark subtraction
    for expo, arr in exposure_dict.items():
        exposure_dict[expo] = arr-mean_dark_dict[expo]
    
    #Transfer curve
    transfer_processor(exposure_dict)
    
    #Linearity curve
    mean_dict = mean_iterator(exposure_dict)
    
    linearity_processor(mean_dict)
    
    
Main()