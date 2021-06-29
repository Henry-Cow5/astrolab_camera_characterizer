"""
@author: Jimmy Ardoin
"""

import os
import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
                
        image_arr_list = [np.array(Image.open(file)) for file in image_list]
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
    return (exposure_dict, dark_dict)

def mean_iterator(dark_dictionary):
    """
    Iterates through the dictionary of sub-stacks to give back a dictionary of mean images.
    Then filters through images with sigma clpping to reject outliers.
    
    Parameters
    ----------
    dark_dictionary : dict
        A dictionary such that each value is a subarray of stack of a certain
        exposure key.

    Returns
    -------
    mean_dict : dict
        A dictionary of the mean images of dark_dictionary sub_arrays with the
        exposures as keys.

    """
    
    print('Creating mean dark images')
    
    #Iterating through dictionary for mean
    mean_dict = {}
    mean_dict_unfil={}
    for exp, sub_arr in dark_dictionary.items():
        mean_dict_unfil[exp] = np.mean(sub_arr, axis=0)
        
    #Filtering results with sigma clipping
    print('Sigma clipping mean dark images')
    clip_criterion = 5
    for exp, sub_arr in mean_dict_unfil.items():
        median = np.median(sub_arr)
        stdev = np.std(sub_arr)
        sub_arr = np.where(sub_arr > (median + stdev*clip_criterion), median, sub_arr)
        sub_arr = np.where(sub_arr < (median - stdev*clip_criterion), median, sub_arr)
        mean_dict[exp] = sub_arr
    return(mean_dict)

def regressor(rms_list, mean_list, var_list):
    
    print('Regressing data')
    index = np.zeros(mean_list[0].shape)
    gain = np.empty(mean_list[0].shape)
    read_noise = np.empty(mean_list[0].shape)
    
    mean_stack = np.stack(mean_list)
    var_stack = np.stack(var_list)
    
    for i, j in np.argwhere(index==0):
        means = mean_stack[:,i,j]
        varis = var_stack[:,i,j]
        gain[i,j], read_noise[i,j] = np.polyfit(means, varis, deg = 1)
        
    return(gain, read_noise)

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
        
    #Performing regression
    gain, read_noise = regressor(rms_list, mean_list, var_list)
    
    #Plotting results
    print('Now plotting results')
    fig1 = plt.figure()
    plt.title('Gain Map')
    ax1 = fig1.add_subplot(111)
    cax = ax1.matshow(gain, cmap='plasma')
    fig1.colorbar(cax, label='Gain (electrons/ADU)')
    plt.text(10,80,'Average gain :' + str(np.mean(gain))[0:5])
    plt.show()
    plt.close()
    
    median = np.median(gain)
    stdev = np.std(gain)
    gain = np.where(gain > median + stdev*4, median, gain)
    
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
    cax = ax2.matshow(read_noise, cmap='Greys')
    fig2.colorbar(cax, label='ADU')
    plt.show()
    plt.close()
    
    return

def Main():
    
    exposure_dict, dark_dict = tiff_getter()
    mean_dict = mean_iterator(dark_dict)
            
    #Dark subtraction
    for expo, arr in exposure_dict.items():
        exposure_dict[expo] = arr-mean_dict[expo]
    
    transfer_processor(exposure_dict)

Main()
    
    
   