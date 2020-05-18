"""
This module classes that generate and apply random affine transformations to images and point.

For example, suppose we want to rotate an image and a point randomly
between +/-10 degrees and translate it +/-20 pixels in the x axis and +/-30
pixels in the y axis.

Ths can be done as follows:

Import module
>>> import augment as aug

Create a new function that will randomly rotate +/-10 degrees
>>> R = aug.Rotate([-10,10])

Create a new function that will randomly translate +/-20 in x and +/-30 in y
>>> T = aug.Translate([-20,20], [-30, 30])

Create a new function that combines all of the listed function in the specified order
>>> tran = aug.Pipeline([R, T])

Generate an 3x3 affine transfrom matrix
>>> tran.generate()

Apply transform to image
>>> img_t = tran.transform_image(img)

Apply transform to point
>>> pt_t = trans.transform_point(pt)

"""
import numpy as np
import cv2
from abc import ABC, abstractmethod


class RandomAffineTransform(ABC):
    """
    Base class for all random affine transform classes. Derived classes must
    implement the abstract generate method.

    Attributes
    ----------
    _T : 3x3 ndarray
        Affine transformation matrix

    Property
    --------
    transform : 3x3 ndarray
        Getter property to get access to transforamtion matrix_T

    Methods
    -------
    generate()
        Abstract method to generate random affine matricies by derived classes.

    _urand(val_range) : float
        Returnes a random number sampled from uniform distribution with range
        defined by val_range.

    transform_image(img) : ndarray
        Applies affine transform to an image and returns transformed image

    transorm_point(pt, rc_order=False) : (float, float)
        Applies affine transform to a point (x, y) (default) or (r, c) (when
        rc_order==True) and returns transformed image.

    """
    def __init__(self):
        ABC.__init__(self)
        self._T = np.zeros((3,3), dtype=np.float)
        self.generate()

    @abstractmethod
    def generate(self):
        """ Abstract method. Must be defined by derived classes and must generate
        a random affine transformation matrix and store it in self._T. This
        transformation matrix will be used to transform images and points.
        """
        pass

    def _urand(self, val_range):
        """
        Returnes a random number sampled from uniform distribution with range
        defined by val_range.

        Parameters
        ----------
        val_range: float or array_like of floats. First and last element are
            used to specify the bounds of the distribution.

        Returns
        -------
        out: float or integer drawn from uniform distribution

        Examples
        --------
        Sample from a uniform distribution with range [0, 5)
        >>> rand_val([0, 5])
        >>> 3.2
        OR
        >>> rand_val([0, 3, 5]) # Only looks at first and last values
        >>> 1.2
        If range is not specified, return whatever is passed in (sample from discrete value)
        >>> rand_val([5,])
        >>> 5.0
        OR
        >>> rand_val(5.0)
        >>> 5.0
        """
        if isinstance(val_range, (list, tuple, np.ndarray)):
            return np.random.uniform(val_range[0], val_range[1])
        # Assume x is a number
        return val_range

    def transform_image(self, img, border_val=0):
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
        return cv2.warpAffine(src=img, M=self._T[:2], dsize=(ncols, nrows), borderValue=border_val)

    def transform_point(self, pt, rc_order=False):
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
        tpt = np.matmul(self._T, tpt)
        return (tpt[i0], tpt[i1])

    @property
    def transform(self):
        """ Getter for affine transform matrix self._T. """
        return self._T


class Translate(RandomAffineTransform):
    """
    Creates an object that can generate random translation matricies and
    apply them to images and points. Translation values are sampled from
    uniform distribution with specified range.

    Parameters
    ----------
    {x|y}_range: float or array_like of floats. First and last element are used
        to specify the bounds of the distribution of the respective dimension
    """
    def __init__(self, x_range=0, y_range=0):
        # TODO: do data type checks here?
        self.x_range = x_range
        self.y_range = y_range
        RandomAffineTransform.__init__(self)

    def generate(self):
        """ Generates a random 3x3 ndarray translation matrix. """
        x = self._urand(self.x_range)
        y = self._urand(self.y_range)
        self._T = np.array(((1, 0, x),
                            (0, 1, y),
                            (0, 0, 1)), dtype=np.float)

class Scale(RandomAffineTransform):
    """
    Creates an object that can generate random scale matricies and
    apply them to images and points. Scale values are sampled from
    uniform distribution with specified range.

    Parameters
    ----------
    {x|y}_range: float or array_like of floats. First and last element are used
        to specify the bounds of the distribution of the respective dimension

    Returns
    -------
    out: 3x3 ndarray
    """
    def __init__(self, x_range=1, y_range=1):

        self.x_range = x_range
        self.y_range = y_range
        RandomAffineTransform.__init__(self)

    def generate(self):
        """ Generates a random 3x3 ndarray scale matrix. """
        x = self._urand(self.x_range)
        y = self._urand(self.y_range)
        self._T =  np.array(((x, 0, 0),
                             (0, y, 0),
                             (0, 0, 1)), dtype=np.float)

class Flip(RandomAffineTransform):
    """
    Creates an object that can generate random flip/mirror matricies and
    apply them to images and points. Binary flip values for each axis are 
    sampled from a Bernoulli distribution.

    Parameters
    ----------
    {x|y}_flip: Boolean value to enable or disable random flipping.
    """
    def __init__(self, flip_x=False, flip_y=False):
        self.flip_x = flip_x
        self.flip_y = flip_y
        RandomAffineTransform.__init__(self)

    def generate(self):
        """ Generates a random 3x3 ndarray flip/mirror matrix. """
        x, y = 1, 1
        if self.flip_x:
            x = np.random.choice((-1, 1))
        if self.flip_y:
            y = np.random.choice((-1, 1))
        self._T =  np.array(((x, 0, 0),
                             (0, y, 0),
                             (0, 0, 1)), dtype=np.float)

class Rotate(RandomAffineTransform):
    """
    Creates an object that can generate random rotation matricies and
    apply them to images and points. rotation value is sampled from a
    uniform distribution with specified range.

    Parameters
    ----------
    angle_range: float or array_like of floats. First and last element are used
        to specify the bounds of the distribution
    """
    def __init__(self, angle_range=0):
        
        self.a_range = angle_range
        RandomAffineTransform.__init__(self)

    def generate(self):
        """ Generates a random 3x3 ndarray scale matrix. """
        a = self._urand(self.a_range)*np.pi/180.0
        c = np.cos(a)
        s = np.sin(a)
        self._T =  np.array((( c, s, 0),
                             (-s, c, 0),
                             ( 0, 0, 1)), dtype=np.float)
    
class Shear(RandomAffineTransform):
    """
    Creates an object that can generate random shear matricies and
    apply them to images and points. Shear angle is sampled from
    uniform distribution with specified range and applied to the specified axis.

    Parameters
    ----------
    angle_range: float or array_like of floats. First and last element are used
        to specify the bounds of the distribution
    axis: 0 or 1 to specify shear direction (x and y respectively)
    """
    def __init__(self, angle_range=0, axis=0):
        self.a_range = angle_range
        self.axis = axis
        RandomAffineTransform.__init__(self)

    def generate(self):
        """ Generates a random 3x3 ndarray shear matrix. """
        a = self._urand(self.a_range)*np.pi/180.0
        x, y = 0, 0
        if self.axis == 0:
            x = np.tan(a)
        elif self.axis == 1:
            y = np.tan(a)
            
        self._T =  np.array(((1, x, 0),
                             (y, 1, 0),
                             (0, 0, 1)), dtype=np.float)

class Pipeline(RandomAffineTransform):
    """
    Creates an object that can generate random transformation matricies by combining
    random affine transformations from a list.
    
    Parameters
    ----------
    transform_list: list or tuple of functions which generate a random transform
        when called
    """
    def __init__(self, rand_transform_list):

        self.rand_transform_list = rand_transform_list
        RandomAffineTransform.__init__(self)
        
    def generate(self):
        """ Generates a random 3x3 ndarray matrix. """
        self._T = np.eye(3, dtype=np.float)
        for tran in self.rand_transform_list:
            tran.generate()
            self._T = np.matmul(tran.transform, self.transform)



if __name__ == '__main__':
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    colors = (RED, GREEN, BLUE, WHITE)
    
    # Create a new images
    img0 = np.zeros((400, 400, 3), dtype=np.uint8)
    img1 = np.zeros((400, 400, 3), dtype=np.uint8)

    square = ((150, 150), (250, 150), (250, 250), (150, 250), (150, 150))
    for i in range(4):
        img0 = cv2.line(img0, pt1=square[i], pt2=square[i+1], color=colors[i])
        

    # Draw circle on image 2
    img1 = cv2.circle(img1, (200, 200), 50, (255, 0, 0), 3)

    # Define some points
    pts = [(250, 150),
           (250, 250),
           (150, 250),
           (150, 150),]
    
    # Create Transform Pipeline
    rand_T  = Pipeline(
        (Translate(-200, -200),
         # Flip(True, True),
         Rotate((-30, 30)),
         Shear((-30, 30), axis=0),
         Shear((-30, 30), axis=1),
         Scale((0.5, 3), (0.5, 3)),
         Translate(200, 200),
         Translate((-50, 50), (-50, 50)),))

    # Generate a new transformation matrix
    rand_T.generate()

    # Apply transformation matrix to list of images and points
    imgs_out = [rand_T.transform_image(img0), rand_T.transform_image(img1)]
    pts_out = [rand_T.transform_point(pt) for pt in pts] 

     # Draw points on original image 0
    for pt in pts:
        cv2.circle(img0, pt, 3, (255,0,255), 3)
        cv2.circle(img1, pt, 3, (255,0,255), 3)

    # Draw transformed points on transformed image
    for pt in pts_out:
        cv2.circle(imgs_out[0], ((int(pt[0]), int(pt[1]))), 3, (255,0,255), 3)
        cv2.circle(imgs_out[1], ((int(pt[0]), int(pt[1]))), 3, (255,0,255), 3)

    # Show all figures
    cv2.imshow('Image0', img0)
    cv2.imshow('Image1', img1)
    cv2.imshow('Transformed0', imgs_out[0])
    cv2.imshow('Transformed1', imgs_out[1])


    
    cv2.waitKey(0)

    cv2.destroyAllWindows()
