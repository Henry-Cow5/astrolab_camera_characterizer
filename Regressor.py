"""
@author : Jimmy Ardoin
"""
import numpy as np
import sys

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