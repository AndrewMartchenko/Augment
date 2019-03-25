"""
This module contains functions that create new functions which generate
affine transformation matricies.
For example, suppose we want to rotate an image and a point randomly
between +/-10 degrees and translate it +/-20 pixels in the x axis and +/-30
pixels in the y axis.

Ths can be done as follows:

Import module
>>> import augment as aug

Create a new function that will randomly rotate +/-10 degrees
>>> r_rotation = rotate([-10,10])

Create a new function that will randomly translate +/-20 in x and +/-30 in y
>>> r_translate = translate([-20,20], [-30, 30])

Create a new function that combines all of the listed function in the specified order
>>> gen = combine([r_rotation, r_translate])

Generate an 3x3 affine transfrom matrix
>>> T = gen()

Apply transform to image
>>> img_t = transform_image(T, img)

Apply transform to point
>>> pt_t = transform_point(T, pt)

"""

import numpy as np
import cv2

def rand_val(val_range):
    """
    Returnes a random number specified by the range in val_range

    Parameters
    ----------
    val_range: float or array_like of floats. First and last element are
        used to specify the bounds of the distribution.

    Returns
    -------
    out: float or integer drawn from uniform distribution

    Examples
    --------
    Returns value sampled from a uniform distribution with range [0, 5)
    >>> rand_val([0,5])
    >>> 3.2
    Returns value from range [0, 10) as bound is defined by first and last element
    >>> rand_val((0,5,10))
    >>> 7.8
    Returns 5.0 as bound is defined by first and last element
    >>> rand_val([5,])
    >>> 5.0
    Returns whatever is passed in
    >>> rand_val(5.0)
    >>> 5.0
    """
    if isinstance(val_range, (list, tuple, np.ndarray)):
        return np.random.uniform(val_range[0], val_range[-1])
    # Assume val_range is a number
    return val_range

def return_function(func):
    """
    Decorator to wrap functions with lambda expression.
    """
    def wrapper(*args, **kwargs):
        """ Wraps a lambda expression around func and it's arguments """
        return lambda: func(*args, **kwargs)
    return wrapper

@return_function
def translate(x_range=0, y_range=0):
    """
    Creates a function that randomly generates translation matricies. Translation
    values are sampled from uniform distribution with specified range.

    Parameters
    ----------
    {x|y}_range: float or array_like of floats. First and last element are used
        to specify the bounds of the distribution of the respective dimension

    Returns
    -------
    out: 3x3 ndarray
    """
    x = rand_val(x_range)
    y = rand_val(y_range)
    return np.array(((1, 0, x),
                     (0, 1, y),
                     (0, 0, 1)), dtype=np.float)


@return_function
def scale(x_range=1, y_range=1):
    """
    Creates a function that randomly generates scale matricies. Scale values are
    sampled from uniform distribution with specified range.

    Parameters
    ----------
    {x|y}_range: float or array_like of floats. First and last element are used
        to specify the bounds of the distribution of the respective dimension

    Returns
    -------
    out: 3x3 ndarray
    """
    x = rand_val(x_range)
    y = rand_val(y_range)
    return np.array(((x, 0, 0),
                     (0, y, 0),
                     (0, 0, 1)), dtype=np.float)

@return_function
def flip(flip_x=False, flip_y=False):
    """
    Creates a function that randomly generates flip/mirror matricies. Axis flip
    is sampled from a Bernoulli distribution.

    Parameters
    ----------
    {x|y}_flip: Boolean value to enable or disable random flipping.

    Returns
    -------
    out: 3x3 ndarray
    """  
    x, y = 1, 1
    if flip_x:
        x = np.random.choice((-1,1))
    if flip_y:
        y = np.random.choice((-1,1))
    return np.array(((x, 0, 0),
                     (0, y, 0),
                     (0, 0, 1)), dtype=np.float)

@return_function
def rotate(angle_range=0):
    """
    Creates a function that randomly generates rotation matricies. Rotation angle is
    sampled from uniform distribution with specified range.

    Parameters
    ----------
    angle_range: float or array_like of floats. First and last element are used
        to specify the bounds of the distribution

    Returns
    -------
    out: 3x3 ndarray
    """
    a = rand_val(angle_range)*np.pi/180.0
    c = np.cos(a)
    s = np.sin(a)
    return np.array((( c, s, 0),
                     (-s, c, 0),
                     ( 0, 0, 1)), dtype=np.float)

@return_function
def shear(angle_range=0, axis=0):
    """
    Creates a function that randomly generates shear matricies. Shear angle is
    sampled from uniform distribution with specified range and axis direction.

    Parameters
    ----------
    angle_range: float or array_like of floats. First and last element are used
        to specify the bounds of the distribution
    axis: 0 or 1 to specify shear direction (x and y respectively)

    Returns
    -------
    out: 3x3 ndarray
    """
    a = rand_val(angle_range)*np.pi/180.0
    x, y = 0, 0
    if axis == 0:
        x = np.tan(a)
    elif axis == 1:
        y=np.tan(a)
    return np.array((( 1, x, 0),
                     ( y, 1, 0),
                     ( 0, 0, 1)), dtype=np.float)

@return_function
def combine(transform_functions):
    ''' 
    Creates a function that randomly generates an affine transfrom from a list
    of transform generator functions.
    
    Parameters
    ----------
    transform_list: list or tuple of functions which generate a random transform
        when called

    Returns
    -------
    out: function that generates random affine transfom.
    '''
    T = np.eye(3, dtype=np.float)
    for tran in transform_functions: # Can reduce function be used for this?
        T = np.matmul(tran(), T)
    return T

def transform_image(T, img):
    ''' 
    Applies affine transform T to an image.
    
    Parameters
    ----------
    T: 3x3 ndarray affine transform matrix
    img: ndarray image data

    Returns
    -------
    out: ndarray transformed image
    '''
    nrows, ncols = img.shape[:2]
    return cv2.warpAffine(img, T[:2], (ncols, nrows))

def transform_point(T, pt, rc_order=False):
    ''' 
    Applies affine transform T to point
    
    Parameters
    ----------
    T: 3x3 ndarray affine transform matrix
    pt: array_like x-y or row-column pair
    rc_order: if True, treats point as row-col order, otherwise x-y order is assumed

    Returns
    -------
    out: array transformed point
    '''
    # Set point index order 
    if rc_order: # if column/row order
        i0, i1 = 1, 0
    else: # if x/y order
        i0, i1 = 0, 1
    tpt = np.array((pt[i0], pt[i1], 1))
    tpt = np.matmul(T, tpt)
    return (tpt[i0], tpt[i1])


if __name__ == '__main__':

    # Create new images
    nrows = 400
    ncols = 600
    img0 = np.zeros((nrows, ncols,3), dtype=np.uint8)
    img1 = np.zeros((nrows, ncols,3), dtype=np.uint8)

    

    # Draw some lines on image 0
    img0 = cv2.line(img0, pt1=(150,150), pt2=(250, 150), color=(255,0,0))
    img0 = cv2.line(img0, pt1=(250,150), pt2=(250, 250), color=(0,255,0))
    img0 = cv2.line(img0, pt1=(250,250), pt2=(150, 250), color=(0,0,250))
    img0 = cv2.line(img0, pt1=(150,250), pt2=(150, 150), color=(255,255,250))

    # Draw circle on image 1
    img1 = cv2.circle(img1, (200, 200), 50, (255,0,0), 3)

    # Define some points
    pts = ((250, 150),(250, 250), (150, 250), (150, 150))

    # List of transform functions
    transform_functions = (
        scale(1/ncols, 1/nrows), # Scale image to unit size
        translate(-0.5, -0.5), # Translate axis to center of image
        flip(True, True),
        rotate((-30,30)),
        shear((-30,30), axis=0),
        shear((-30,30), axis=1),
        scale((0.5, 1.5), (0.5,1.5)),
        translate((-0.1, 0.1), (-0.1,0.1)),
        translate(0.5, 0.5), # Translate image back to top left corner
        scale(ncols, nrows), # Undo unit scaling
    )

    # Combine transform functions into one transform generator function
    gen = combine(transform_functions)
    
    # Generate a new transformation matrix
    T = gen()
    
    # Apply transformation matrix to list of images and points
    imgs_out = [transform_image(T, img0), transform_image(T, img1)]
    pts_out = [transform_point(T, pt) for pt in pts]                           

    # Draw original points on original images
    for i, pt in enumerate(pts):
        cv2.circle(img0, pt, 3+3*i, (255,0,255), 2)
        cv2.circle(img1, pt, 3+3*i, (255,0,255), 2)

    # Draw transformed points on transformed image
    for i, pt in enumerate(pts_out):
        cv2.circle(imgs_out[0], ((int(pt[0]), int(pt[1]))), 3+3*i, (255,0,255), 2)
        cv2.circle(imgs_out[1], ((int(pt[0]), int(pt[1]))), 3+3*i, (255,0,255), 2)

    # Show all figures
    cv2.imshow('Image0', img0)
    cv2.imshow('Image1', img1)
    cv2.imshow('Transformed0', imgs_out[0])
    cv2.imshow('Transformed1', imgs_out[1])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
