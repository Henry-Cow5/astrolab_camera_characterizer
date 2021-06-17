"""
@author: Jimmy Ardoin
Warning: This program will not run on 32-bit systems unless you are looking at
very small files. You need quite a bit of free space for the memory mapping as
well. For 1000 2048x2048 16 bit images, you need 32 GiB of free space, and the
total necessary space should scale linearly to those dimensions. These files
are deleted after the program runs.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
#from matplotlib import cm
import os
import os.path
import sys
#import pandas as pd
#from joypy import joyplot

gain_dict = {'CSC-00085 12': 0.69, 'CSC-00085 16': 1.41, 'CSC-00542 12': 0.61,
             'CSC-00542 16': 1.26}

def fits_getter ():
    """
    Prompts user for directory and gets .fits files in that directory.

    Returns
    -------
    fits_list : list
        The list of .fits file names in the gotten directory
    path : str
        Path name
    dirs : str
        Directory name. Might need this later?

    """

    while True:
        inpt = input('If you are examining a specific directory, enter \'d\'; if you are examining the directory this program is in, enter \'t\'; press \'x\' to quit.')
        fits_list=[]
        if inpt == 't':
            path, dirs, files = next(os.walk("."))
            for file in files:
                if file.endswith('.fits'):
                    fits_list.append(file)
            print('Found ' + str(len(fits_list)) + ' .fits files in this directory.')
            return (fits_list, path, dirs)
        elif inpt == 'd':
            Path = input('Enter exact path to the directory')
            path, dirs, files = next(os.walk(Path))
            for file in files:
                if file.endswith('.fits'):
                    fits_list.append(file)
            print('Found ' + str(len(fits_list)) + ' .fits files in this directory.')
            return (fits_list, path, dirs)
        elif inpt == 'x':
            sys.exit()
        else:
            print('Oops! That was not a valid letter. Try again...')

def fits_info(fits_list, path, gain_dict):
    """
    Gets necessary info from the .fits header.

    Parameters
    ----------
    fits_list : str
        The list of .fits file names in the gotten directory
    path : str
        Path name

    Returns
    -------
    dim_x, dim_y : int
        The dimensions of the pixel in the x and y directions
    exposure : list
        The list of exposure times of the .fits files
    gain : float
        The gain for converting units
    """
    #list of fits header get
    hdu_list = [fits.getheader(path +'/' + str(fit)) for fit in fits_list]
    exposure = []
    dim_x = 0
    dim_y = 0
    for count, hdu in enumerate(hdu_list):
        #There seems to be a few standards for headers. We try these exceptions
        #to catch all the names we have found.
        try:
            exposure = np.append(exposure, hdu['exposure'])
        except KeyError:
            exposure = np.append(exposure, hdu['exptime'])
        #getting size and gain of images
        if count == 0:
            dim_x = hdu['naxis1']
            dim_y = hdu['naxis2']
            gain = gain_dict[str(hdu['hierarch serialnumber']) + ' ' + str(hdu['bitpix'])]
    return (int(dim_x), int(dim_y), exposure, gain)

def stacker(path, fits_list, gain, dim_x, dim_y):
    """
    Reads in image data and stacks them into a memory-mapped array

    Parameters
    ----------
    path : str
        Path name
    fits_list : list
        The list of .fits file names in the gotten directory.
    gain : float
    
    dim_x, dim_y : ints
        number of pixels in x and y.

    Returns
    -------
    stack : 3D memmapped numpy array (n, p, p) n is number of images, and p is pixels
        The stack of images together.

    """
    
    print("Reading in images")
    image_concat = [fits.getdata(path + '/' + str(fit)) for fit in fits_list]
    print("Stacking images")
    #We have to remove stack if the program crashed earlier
    if os.path.exists('stack.memmap'):
        os.remove('stack.memmap')
    #The stack is stored written to disk in the same folder the program is
    #located in
    stack = np.memmap('stack.memmap', dtype='float64', mode='w+',
                            shape=(len(image_concat), dim_y, dim_x))
    #Because of how numpy arrays work, we have to manually assign contents
    #to the memmapped arrays, so as to not create copies which would destroy 
    #our memory.
    for count, image in enumerate(image_concat):
        stack[count] = image
    stack_converter(stack, gain)
    return (stack)

def stack_converter(stack, gain):
#Converts the stack from ADU to electrons
    print('Converting to electrons. This will take a while')
    if os.path.exists('mean.memmap'):
        os.remove('mean.memmap')
    mean = np.memmap('mean.memmap', dtype='float64', mode ='w+',
                     shape=stack.shape[1:])
    np.copyto(mean, np.mean(stack, axis=0))
    for i in range(stack.shape[0]):
        stack[i] = (stack[i]-mean)*gain
    #Removing the mean memmap
    mean._mmap.close()
    del mean
    os.remove('mean.memmap')
    return(stack)
        
def set_order_preserve(list):
    """
    Eliminates duplicates in a list while preserving the original order of the 
    list.

    Parameters
    ----------
    list : list

    Returns
    -------
    list: list
        The original list in the same order but with duplicates eliminated

    """
    seen = set()
    seen_add = seen.add 
    return [x for x in list if not (x in seen or seen_add(x))]

def dark_dictionary_maker(stack, exposure):
    """
    Splits the stack along the frames such that each element of dark_dictionary
    is a subarray of stack with a unique exposure key. 

    Parameters
    ----------
    stack : 3D numpy array (n, p, p) n is number of images, and p is pixels
        The stack of images together.
    column_number : int
        The largest power of two the image dimensions can be evenly divided by.
        If this is 1 it gives an error for odd pixel dimensions.
    exposure : list
        The list of exposure times of the .fits files

    Returns
    -------
    dark_dictionary : dict
        A dictionary such that each value is a subarray of stack with a unique
        exposure key.

    """
    
    exposure_order = set_order_preserve(exposure)
    split_index = 0
    dark_dictionary = {}
    #Creating a dictionary to keep track of list of pixels paired with index.
    for exp in exposure_order:
        count = np.count_nonzero(exposure==exp)
        dark_dictionary[exp] = stack[split_index:(split_index+count)]
        split_index += count
    return(dark_dictionary)
    
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
    
    print('Creating mean images')
    #Iterating through dictionary for mean
    mean_dict = {}
    mean_dict_unfil={}
    for exp, sub_arr in dark_dictionary.items():
        mean_dict_unfil[exp] = np.mean(sub_arr, axis=0)
    #Filtering results with sigma clipping
    clip_criterion = 5
    print('Filtering results')
    for exp, sub_arr in mean_dict_unfil.items():
        median = np.median(sub_arr)
        stdev = np.std(sub_arr)
        sub_arr = np.where(sub_arr > (median + stdev*clip_criterion), median, sub_arr)
        sub_arr = np.where(sub_arr < (median - stdev*clip_criterion), median, sub_arr)
        show_array(sub_arr)
        mean_dict[exp] = sub_arr
    return(mean_dict)

def show_array(array):
#Shows and closes an array with colorbar
    plt.matshow(array)
    plt.colorbar()
    plt.show()
    plt.close()

def stdev_reporter(stdev, median, noise_criteria):
    """
    Shows the histogram of the standard deviation values across all pixels and
    limits of the noise_criteria bins. Also corrects the final bin value.

    Parameters
    ----------
    stdev : array (pxp)
        Standard deviation image
    median : float
        Median standard dewviation value
    noise_criteria : list
        list of list pairs which delimit the bins for noise analysis

    Returns
    -------
    noise_criteria : list
        Same as above, but last pair has been corrected to the maximum value of the set.

    """
    
    #standard deviation histogram
    print('stdev Min:', np.min(stdev))
    print('stdev Max:', np.max(stdev))
    print('stdev Median:', + median)
    print('stdev rms:', np.sqrt(np.mean(stdev**2)))
    plt.title('stdev pixel value histogram')
    plt.xlabel('Number of electrons')
    plt.ylabel('Number of pixels')
    y, _, _ = plt.hist(stdev.flatten(), bins='auto')
    #noise bins
    for count, pair in enumerate(noise_criteria):
        if pair[1]==0:
            #Replacement of last bin size
            pair[1] = np.max(stdev)/median
        plt.axvline(pair[0]*median, color='red')
        plt.axvline(pair[1]*median, color='red')
        plt.text((pair[0]*median+pair[1]*median)/2, y.max(), 'Bin ' + str(count+1),
                 ha='center')
    plt.show()
    plt.close()
    return(noise_criteria)

def noise_check(stack_std, median_std, dim_x, dim_y, noise_criteria):
    """
    Checks the standard deviation image for elements with a standad deviation 
    greater than noise_criterion times the median standard deviation, and 
    returns an array of p x p with the noisy stdev values and 0sin every other
    spot.

    Parameters
    ----------
    stack_std : 2D numpy array (p,p)
        The image of the standard deviation of the pixels across all layers.
    median_std : float64
        The median standard deviation value
    dim_x : int
        Number of x pixels
    dim_y : int
        Number of y pixels
    noise_criteria : list
        list of list pairs which delimit the bins for noise analysis

    Returns
    -------
    noisy_stds : list of 2D numpy arrays (p,p)
        The same images as stack_std, but any non-selected pixels are 0.
    smallest_bin : int
        The number of pixels in the smallest bin (usually the last one)

    """
    
    #list of lists initialization
    noisy_stds = [[] for i in range(len(noise_criteria))]
    #Starts with the first bin in noise_criteria
    for count, pair in enumerate(noise_criteria):
        temp_count = 0
        noisy_temp = []
        count_list = []
        #For each pair creates a noisy_temp 1d array which will be reshaped into
        #2d image of only items in that bin.
        for std in stack_std.flatten():
            if std > pair[0]*median_std and std <= pair[1]*median_std:
                noisy_temp.append(std)
                temp_count+=1
            else:
                noisy_temp.append(0)
        noisy_stds[count] = noisy_temp
        print(str(temp_count) + ' pixels found with between ' \
            + str(noise_criteria[count]) + ' median standard deviations. That' \
            ' is ~' + '%.3f' % (temp_count/len(stack_std.flatten())*100) + '% of pixels.')
        count_list = np.append(count_list, temp_count)
    smallest_bin = min(count_list)
    noisy_stds = [np.reshape(arr, (dim_x, dim_y)) for arr in noisy_stds]
    return (noisy_stds, int(smallest_bin))

def noise_getter(noisy_stds, stack, count, noise_criteria, smallest_bin):
    """
    Makes a dictionary pairing the indices of selected pixels with the list of
    their values across frames up to smallest_bin

    Parameters
    ----------
    noisy_stds : 2D numpy array (p,p)
        The same image as stack_std, but any non-noisy pixels are 0.
    stack : 3D numpy array (n, p, p) n is number of images, and p is pixels
        The stack of images together.
    noise_criteria : list
        list of list pairs which delimit the bins for noise analysis
    smallest_bin : int
        The size of the smallest bin.
    Returns
    -------
    noisy_pixel_dict : dict
        The pixel index of each noisy pixel paired with the list of that 
        pixel's values across all layers.'

    """
    print('Isolating pixels for bin between ' + str(noise_criteria[count]) + 
          ' median standard deviations')
    noisy_std_indices = np.argwhere(noisy_stds)
    #random sort to get a representative sample
    np.random.shuffle(noisy_std_indices)
    indices_list = noisy_std_indices.tolist()
    
    pixel_dict = {}
    count = 0
    for index_1, index_2 in indices_list:
        index = str(str(index_1) + ', ' + str(index_2))
        pixel_dict[index] = [stack[frame, index_1, index_2] for frame in \
                                   range(0, stack.shape[0])]
        count += 1
        if count == smallest_bin:
            return (pixel_dict)
        
    return (pixel_dict)

def regressor(mean_dictionary, dim_x, dim_y):
    """
    Regresses on a pixel by pixel basis each exposure's mean frame with that 
    exposure time and returns the dark current image and the bias image.

    Parameters
    ----------
    mean_dict : dict
        A dictionary of the mean images of dark_dictionary sub_arrays with the
        exposures as keys.
    dim_x : int
        Number of x pixels
    dim_y : int
        Number of y pixels

    Returns
    -------
    dark_curr : 2d array (pxp)
        The image of dark current values for each pixel.
    bias : 2d array (pxp)
        The image of bias values for each pixel.

    """
    
    print('Regressing each pixel for dark current')
    exposures = list(mean_dictionary.keys())
    #Stacks mean frames per exposure time
    mean_stack = np.stack(mean_dictionary.values())
    dark_curr = np.zeros((dim_y, dim_x))
    bias = np.zeros((dim_y, dim_x))
    #We create a (pxp) 0 array to get the list of indices to iterate 
    #through these images
    ###For future maintainers: this is definitely not the most efficient way
    ###to perform this operation. I invite you to try and improve it.
    index = np.zeros((dim_y,dim_x))
    for i, j in np.argwhere(index==0):
        means = mean_stack[:,i,j]
        dark_curr[i,j], bias[i,j] = np.polyfit(exposures, means, deg = 1)

    print('Average dark current is: ' + str(np.mean(dark_curr.flatten())))
    print('The median dark current is: ' + str(np.median(dark_curr.flatten())))
    print('The standard deviation of the dark current is: ' + str(np.std(dark_curr.flatten())))
    show_array(dark_curr)
    return dark_curr, bias
        
def pixel_hist_maker(noisy_pixel_dict, count, noise_criteria, mini, maxi):
#Creates histograms for each noisy pixel, a complete layered step histogram,
#and a scatter plot of the noisy pixels.

#Creates bin_num number of bins between the smallest and largest values of all
#noise pixels.
    bin_num = 40
    bins = np.linspace(mini, maxi, bin_num)
#Creates individual histograms
    #for index, hist_pool in noisy_pixel_dict.items():
        #indx = [int(x) + 1 for x in index.split(',')]
        #plt.title('Pixel #(' + ', '.join(map(str, indx)) + ')')
        #plt.hist(hist_pool, bins)
        #plt.show()
        #plt.close()
#Creates complete histogram
    print('Plotting data for pixels between ' + str(noise_criteria[count])
              + ' median standard deviations')
    if str(noise_criteria[count])[1:7].endswith(']'):
        plt.title('Pixel histogram for pixels within (' + str(noise_criteria[count])[1:7]
              + ' median standard deviations')
    else:
        plt.title('Pixel histogram for pixels within (' + str(noise_criteria[count])[1:7]
              + '] median standard deviations')
    plt.hist(noisy_pixel_dict.values(), bins, histtype='step', linewidth=0.5)
    plt.xlabel('Number of electrons')
    plt.ylabel('Number of pixels')
    plt.show()
    plt.close()

#Creates scatter plot of last bin
    if count == len(noise_criteria)-1:
        for index,_ in noisy_pixel_dict.items():
            indx = [int(x) + 1 for x in index.split(',')]
            plt.scatter(indx[0], indx[1], color='blue', s=0.6)
        plt.title('Map of pixels between (' + str(noise_criteria[count])[1:7]
                  + '] median standard deviations')
        plt.gca().set_aspect('equal')
        plt.show()
        plt.close()
    return bins

def joy_maker(dict_list, noise_criteria, bins):
    # # joy_df = pd.DataFrame()
    # # for count, dit in enumerate(dict_list):
    # #     joy_df['bin'] = [str(noise_criteria[count]) for i in \
    # #              range(len(list(dit.values())))]
    # #     joy_df['val' + str(count)] = list(dit.values())
    # new_dict = {}
    # for bin, dit in enumerate(dict_list):
    #     new_dict['bin ' + str(bin+1)] = [item for sublist in dit.values() for item in sublist]
    # # print(joy_df)
    # plt.figure()
    # # joyplot(data=joy_df,
    # #     by='bin',
    # #     column=['val' + str(count) for count in range(len(dict_list))])
    # joyplot(new_dict, linecolor='black', hist=True, bins=bins, colormap=cm.autumn_r)
    # plt.show()
    # plt.close()
    fig = plt.figure()
    fig.set_size_inches(11, 17)
    fig.set_dpi(100)
    iterator=len(dict_list)*100+11
    for dit in dict_list:
        if dit==dict_list[0]:
            ax1=plt.subplot(iterator)
        else:
            plt.subplot(iterator, sharex=ax1, sharey=ax1)
        plt.hist(dit.values(), bins, histtype='step', linewidth=0.5)
        iterator += 1
    plt.xlabel('Electrons')
    plt.show()
        
def bias_run(stack, dim_x, dim_y, noise_criteria):
#Main body of bias noise analysis
    print('Creating standard deviation image')
    mini = np.min(stack)
    maxi = np.max(stack)
    #Calculates standard deviation image
    stack_std = np.std(stack, axis=0)
    show_array(stack_std)
    median_std = np.median(stack_std)
    noise_criteria = stdev_reporter(stack_std, median_std, noise_criteria)
    print('Now checking for noisy pixels')
#Noise analysis starts here
    noisy_stds, smallest_bin = noise_check(stack_std, median_std, dim_x, dim_y,
                                           noise_criteria)
    print('Smallest bin was found to be ' + str(smallest_bin) + '. This will be' +
      ' the representative sample size for plotting. The samples of larger bins' +
      ' are picked randomly to ensure they are representative.')
    print('Running noise analysis on individual pixels for each bin. This process will take a while')
    #noisy_stds is a list of 2d arrays, so we iterate through each array which
    #corresponds to its own bin.
    dict_list = []
    for count, noisy_arr in enumerate(noisy_stds):
        noisy_pixel_dict = noise_getter(noisy_arr, stack, count, noise_criteria,
                                        smallest_bin)
        bins = pixel_hist_maker(noisy_pixel_dict, count, noise_criteria, mini, maxi)
        dict_list.append(noisy_pixel_dict)
        del noisy_pixel_dict
    #Removing the stack memmap
    stack._mmap.close()
    del stack
    os.remove('stack.memmap')
    joy_maker(dict_list, noise_criteria, bins)

def dark_run(stack, dim_x, dim_y, exposure):
#Main body of dark current analysis
    
    dark_dictionary = dark_dictionary_maker(stack, exposure)
    mean_dictionary = mean_iterator(dark_dictionary)
    dark_curr, bias = regressor(mean_dictionary, dim_x, dim_y)

def Main_body(gain_dict):
#Main body of code. Follow this to see what it's doing.
    while True:
        fits_list, path, dirs = fits_getter()
        dim_x, dim_y, exposure, gain = fits_info(fits_list, path, gain_dict)
        print('What type of images are these?')
        inpt = input('For bias images enter \'b\'; for dark images enter \'d\'; press \'x\' to quit.')
        if inpt == 'b':
            del exposure
            stack = stacker(path, fits_list, gain, dim_x, dim_y)
            #Here we set the criteria for the bins when sampling bias noise.
            #Last bin value will be corrected for max standard devation. Make
            #sure final value is always 0.
            noise_criteria = [[0,1], [1, 1.5], [1.5,2.5], [2.5, 3.5], [3.5, 4], [4, 0]]
            bias_run(stack, dim_x, dim_y, noise_criteria)
            return
        elif inpt == 'd':
            stack = stacker(path, fits_list, gain, dim_x, dim_y)
            exposure = np.array(exposure.astype(np.float32))
            dark_run(stack, dim_x, dim_y, exposure)
            return
        elif inpt == 'x':
            sys.exit()
        else:
            print('Oops! That was not a valid letter. Try again...')


Main_body(gain_dict)

