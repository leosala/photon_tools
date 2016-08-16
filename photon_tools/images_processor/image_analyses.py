import numpy as np
from time import time
import sys
import h5py
import pydoc


def get_projection(results, temp, image_in, axis=0, thr_hi=None, thr_low=None):
    """Returns a spectra (projection) over an axis of an image. This function is to be used within an AnalysisProcessor
    instance.
    
    Parameters
    ----------
    results : dict
        dictionary containing the results. This is provided by the AnalysisProcessor class
    temp : dict
        dictionary containing temporary variables. This is provided by the AnalysisProcessor class
    image_in : Numpy array
        the image. This is provided by the AnalysisProcessor class
    axis: int, optional
        the axis index over which the projection is taken. This is the same as array.sum(axis=axis). Default is 0
    thr_hi: float, optional
        Upper threshold to be applied to the image. Values higher than thr_hi will be put to 0. Default is None
    thr_low: float, optional
        Lower threshold to be applied to the image. Values lower than thr_low will be put to 0. Default is None
    
    Returns
    -------
    results, temp: dict
        Dictionaries containing results and temporary variables, to be used internally by AnalysisProcessor
    """
    if axis == 1:
        other_axis = 0
    else:
        other_axis = 1

    # static type casting, due to overflow possibility...
    if temp["current_entry"] == 0:
        if temp["image_dtype"].name.find('int') != -1:
            results["spectra"] = np.empty((results['n_entries'], temp["image_shape"][other_axis]), dtype=np.int64) 
        elif temp["image_dtype"].name.find('float') != -1:
            results["spectra"] = np.empty((results['n_entries'], temp["image_shape"][other_axis]), dtype=np.float64) 
        
    # if there is no image, return NaN
    if image_in is None:
        result = np.ones(temp["image_shape"][other_axis], dtype=temp["image_dtype"])
        result[:] = np.NAN
    else:
        image = image_in.copy()
        if thr_low is not None:
            image[image < thr_low] = 0
        if thr_hi is not None:
            image[image > thr_hi] = 0
    
        result = np.nansum(image, axis=axis)

    # if result[result > 1000] != []:
    # print temp['current_entry'], result[result > 1000]
    results["spectra"][temp['current_entry']] = result
    temp["current_entry"] += 1
    return results, temp


def get_mean_std(results, temp, image_in, thr_hi=None, thr_low=None):
    """Returns the average of images and their standard deviation. This function is to be used within an
    AnalysisProcessor instance.
    
    Parameters
    ----------
    results : dict
        dictionary containing the results. This is provided by the AnalysisProcessor class
    temp : dict
        dictionary containing temporary variables. This is provided by the AnalysisProcessor class
    image_in : Numpy array
        the image. This is provided by the AnalysisProcessor class
    thr_hi: float, optional
        Upper threshold to be applied to the image. Values higher than thr_hi will be put to 0. Default is None
    thr_low: float, optional
        Lower threshold to be applied to the image. Values lower than thr_low will be put to 0. Default is None
    
    Returns
    -------
    results, temp: dict
        Dictionaries containing results and temporary variables, to be used internally by AnalysisProcessor
    """
    if image_in is None:
        return results, temp
        
    image = image_in.copy()
    
    if thr_low is not None:
        image[image < thr_low] = 0
    if thr_hi is not None:
        image[image > thr_hi] = 0
    
    if temp["current_entry"] == 0:
        temp["sum"] = np.array(image)
        temp["sum2"] = np.array(image * image)
    else:
        temp["sum"] += image
        temp["sum2"] += np.array(image * image)

    temp["current_entry"] += 1    

    return results, temp    
    

def get_mean_std_results(results, temp):
    """Function to be applied to results of image_get_std_results. This function is to be used within an
    AnalysisProcessor instance, and it is called automatically.
    
    Parameters
    ----------
    results : dict
        dictionary containing the results. This is provided by the AnalysisProcessor class
    temp : dict
        dictionary containing temporary variables. This is provided by the AnalysisProcessor class
    
    Returns
    -------
    results: dict
        Dictionaries containing results and temporary variables, to be used internally by AnalysisProcessor. Result
        keys are 'images_mean' and 'images_std', which are the average and the standard deviation, respectively.
    """
    if not temp.has_key("sum"):
        return results        

    mean = temp["sum"] / temp["current_entry"]
    std = (temp["sum2"] / temp["current_entry"]) - mean * mean
    std = np.sqrt(std)
    results["images_mean"] = mean
    results["images_std"] = std
    return results


def get_histo_counts(results, temp, image, bins=None, roi=None):
    """Returns the total histogram of counts of images. This function is to be used within an AnalysisProcessor
    instance. This function can be expensive.
    
    Parameters
    ----------
    results: dict
        dictionary containing the results. This is provided by the AnalysisProcessor class
    temp: dict
        dictionary containing temporary variables. This is provided by the AnalysisProcessor class
    image: Numpy array
        the image. This is provided by the AnalysisProcessor class
    bins: array, optional
        array with bin extremes        
        
    Returns
    -------
    results, temp: dict
        Dictionaries containing results and temporary variables, to be used internally by AnalysisProcessor
    """
    if image is None:
        return results, temp
        
    if roi is not None:
        image = image[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]

    if bins is None:
        bins = np.arange(-100, 1000, 5)
    t_histo = np.bincount(np.digitize(image.flatten(), bins[1:-1]), 
                          minlength=len(bins) - 1)
    
    if temp["current_entry"] == 0:
        results["histo_counts"] = t_histo
        results["histo_bins"] = bins
    else:
        results["histo_counts"] += t_histo

    temp["current_entry"] += 1
    return results, temp                  
  

def set_roi(image, roi=None):
    """Returns a copy of the original image, selected by the ROI region specified

    Parameters
    ----------
    image: Numpy array
        the input array image
    roi: array
        the ROI selection, as [[X_lo, X_hi], [Y_lo, Y_hi]]
        
    Returns
    -------
    image: Numpy array
        a copy of the original image, with ROI applied
    """
    if roi is not None:
        new_image = image[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
        return new_image
    else:
        return image
 
 
def set_thr(image, thr_low=None, thr_hi=None, replacement_value=0):
    """Returns a copy of the original image, with a low and an high thresholds applied

    Parameters
    ----------
    image: Numpy array
        the input array image
    thr_low: int, float
        the lower threshold
    thr_hi: int, float
        the higher threshold
    replacement_value: int, float
        the value with which values lower or higher than thresholds should be put equal to
        
    Returns
    -------
    image: Numpy array
        a copy of the original image, with ROI applied
    """
    new_image = image.copy()
    if thr_low is not None:
        new_image[new_image < thr_low] = replacement_value
    if thr_hi is not None:
        new_image[new_image > thr_hi] = replacement_value
    return new_image


def subtract_correction(image, sub_image):
    """Returns a copy of the original image, after subtraction of a user-provided image (e.g. dark background)

    Parameters
    ----------
    image: Numpy array
        the input array image
    sub_image: Numpy array
        the image to be subtracted

    Returns
    -------
    image: Numpy array
        a copy of the original image, with subtraction applied
    """
    new_image = image.copy()
    return new_image - sub_image


def mask_pixels(image, mask):
    new_image = image.copy()
    new_image[mask] = 0
    return new_image


def correct_bad_pixels(image, mask, method="swiss"):
    new_image = image.copy()
    points = np.array(np.where(mask is True))
    if method == "swiss":
        # add check if neigh pixels is hot, too
        for x, y in points.T:
            if x == 0 or y == 0 or x == image.shape[0]-1 or y == image.shape[1]-1:
                continue
            new_image[x, y] = new_image[x - 1, y] + new_image[x + 1, y] + new_image[x, y - 1] + new_image[x, y + 1]
            new_image[x, y] /= 4
    return new_image


def roi_bkgRoi(results, temp, image_in, roi, bkg_roi):
    """ Returns the total intensity (sum of all pixels) in a roi and corresponding background based on a background
    roi from an input image. The function checks for overlap between the roi and bkg_roi, and takes it into account.
     This function is to be used within an AnalysisProcessor instance.

    Parameters
    ----------
    results : dict
        dictionary containing the results. This is provided by the AnalysisProcessor class
    temp : dict
        dictionary containing temporary variables. This is provided by the AnalysisProcessor class
    image_in : Numpy array
        the image. This is provided by the AnalysisProcessor class
    roi: Array
        the ROI selection, as [[X_lo, X_hi], [Y_lo, Y_hi]]
    bkg_roi: Array
        the background ROI selection, as [[X_lo, X_hi], [Y_lo, Y_hi]]

    Returns
    -------
    results, temp: dict
        Dictionaries containing intensity in roi and its background and temporary variables, to be used internally by
        AnalysisProcessor
    """

    # check for intersection
    temp_roi = roi + np.array([[-8, 8], [-7, 7]])  # extended roi for safe background intensity
    fintersect = (temp_roi[0][0] < bkg_roi[0][1] and bkg_roi[0][0] < temp_roi[0][1] and
                  temp_roi[1][0] < bkg_roi[1][1] and bkg_roi[1][0] < temp_roi[1][1])

    if fintersect:
        intersect = [[max(temp_roi[0][0], bkg_roi[0][0]), min(temp_roi[0][1], bkg_roi[0][1])],
                     [max(temp_roi[1][0], bkg_roi[1][0]), min(temp_roi[1][1], bkg_roi[1][1])]]
    else:
        intersect = []

    temp_roi = intersect

    img_roi = image_in[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
    img_bkg_roi = image_in[bkg_roi[0][0]:bkg_roi[0][1], bkg_roi[1][0]:bkg_roi[1][1]]
    img_temp_roi = image_in[temp_roi[0][0]:temp_roi[0][1], temp_roi[1][0]:temp_roi[1][1]]

    size_roi = img_roi.shape[0] * img_roi.shape[1]
    size_bkg_roi = img_bkg_roi.shape[0] * img_bkg_roi.shape[1]
    size_temp_roi = img_temp_roi.shape[0] * img_temp_roi.shape[1]

    intensity_roi = sum(sum(img_roi))
    intensity_bkg_roi = sum(sum(img_bkg_roi))
    intensity_temp_roi = sum(sum(img_temp_roi))

    intensity_bkg_roi = (intensity_bkg_roi-intensity_temp_roi) / (size_bkg_roi-size_temp_roi) * size_roi

    if temp["current_entry"] == 0:
        results["intensity"] = np.array(intensity_roi)
        results["bkg"] = np.array(intensity_bkg_roi)
    else:
        results["intensity"] = np.append(results["intensity"], intensity_roi)
        results["bkg"] = np.append(results["bkg"], intensity_bkg_roi)

    temp["current_entry"] += 1

    return results, temp

