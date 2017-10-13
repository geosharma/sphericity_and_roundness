# ! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Determine the sphericity of a particle based on binary 2D projected image.

Directory structure:
Parent directory (foo)
    /data       place the binary image in this directory
    /plots      analysis output will be placed here
    /prog       this script resides in this directory
"""


# import modulues
import numpy as np

# import operation system module os
import os

# import natsort module for natural human sorting
import natsort

# import plotting module
import matplotlib.pyplot as plt

# import module for image manipulation
import cv2


from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import leastsq
import matplotlib
from scipy import odr
from scipy import interpolate as spinter

# Path of R home
os.environ['R_HOME'] = 'C:\Program Files\R\R-3.4.1'
# import rpy2 for R functions, to use loess in R
import rpy2.robjects as ro
from rpy2.robjects import r
import rpy2.robjects.numpy2ri

# to use R for loess fitting from Python
rpy2.robjects.numpy2ri.activate()


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """
    Create new image(numpy array) filled with certain color in RGB.

    The following code has been adapted from stackoverflow member Kimmo's post
    and can be found at the below URL.
    Ref: https://stackoverflow.com/questions/4337902/
                how-to-fill-opencv-image-with-one-solid-color
    """
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def calc_dist(xc, yc, x, y):
    """
    Compute the distance of each 2D points from the center.

    Arguments:
        xc, yc: The center of the circle
        x, y: The points for distance calculation
    Returns:
        2D Euclidean distance between (xc, yc) and (x, y)
    """
    return np.sqrt((x-xc)**2 + (y-yc)**2)


def rloess_smooth(xdata, ydata, xfit, span=0.07, deg=2):
    """
    Use rpy2 to call R's loess function from Pthon

    Input: x, y numpy array of data points
           xfit, numpy array of data at which yfit it to be computed
           lspan, span of the fitting
           ldeg, deg of the polynomial used for fitting

    Output: x, yfit numpy array of x and smoothed y
    """
    # x = ro.FloatVector(list(xdata))
    # y = ro.FloatVector(list(ydata))
    # x_fit = ro.FloatVector(list(xfit))
    rdf = ro.DataFrame({"x": xdata, "y": ydata})
    loess_fit = r.loess('y ~ x', data=rdf, span=span, deg=deg)
    yfit = np.array(list(r.predict(loess_fit, xfit)))
    return yfit


def fillholes(img):
    """
    Fill holes in the sand particles because thresholded images could have some
    holes inside the particles.

    Input: img, thresholded image, binary image

    Ouput: Image with any holes in the sand particle filled. White particle
           with black background.
    """
    # ref: https://www.learnopencv.com/
    # filling-holes-in-an-image-using-opencv-python-c/
    # author: SATYA MALLICK

    # make a copy the thresholded image
    img_floodfill = img.copy()
    # ax.imshow(img_floodfill, cmap=matplotlib.cm.gray, vmin=0, vmax=255)
    # plt.show()
    # fill holes present in the sand particle
    # mask used for flood filling
    # the size needs to be 2 pixels larger than the image
    h, w = img.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    # floodfill from point(10, 50) to avoid the annotation
    cv2.floodFill(img_floodfill, mask, (10, 50), 255)

    # invert the floodfilled image
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)

    # combine the two images to get the foreground with holes filled
    img_out = img | img_floodfill_inv

    return img_out


def particle_countour_params(cnt):
    """
    Derive various partice shape parameters from the boundary trace of the
    particle. Units: pixels

    Input: cnt, the outer most contour (boundary trace) of the particle.

    Output: Dictionary with
            Center of mass, (cX, cY)
            Area, area
            Perimeter, perimeter
            Minimum enclosing circle center and radius, minEnclosingCircle,
            ((mecX, mecY), mecrad)
            Aspect ratio, aspect_ratio
            Diameter: diameter
            Major and min axis of the fitted ellipse: MajMinAxis, (MA, ma)
                (center (x,y), (MA, ma), angle of rotation)
            Minimum area bounding rectangle, minAeaRect,
                (center (x,y), (width, height), angle of rotation)
            Solidity, solidity
    """
    # ref: http://docs.opencv.org/trunk/dd/d49/
    # tutorial_py_contour_features.html

    # find the center of mass from the moments of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h

    # solidity
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area

    # Equivalent Diameter is the diameter of the circle whose area
    # is same as the contour area
    equi_diameter = np.sqrt(4.0 * area/np.pi)

    # min enclosing circle
    perimeter = cv2.arcLength(cnt, True)
    mec = cv2.minEnclosingCircle(cnt)
    # (mecX, mecY), mecrad

    # minimum area bounding rectangle
    rect = cv2.minAreaRect(cnt)

    # fitting an ellipse
    el = cv2.fitEllipse(cnt)

    return {'center_of_mass': (cX, cY), 'area': area, 'perimeter': perimeter,
            'minEnclosingCircle': mec, 'aspect_ratio': aspect_ratio,
            'diameter': equi_diameter, 'ellipse': el,
            'minAreaRect': rect, 'solidity': solidity}


def cart2polar(cnt, center):
    """
    Convert cartesian coordinates to polar coordinates with respect to the
    specified center.
    Input: cnt, [[[x1, y1], [x2, y2], ...]]] contour points
           center, center with respect to which the polar coordinates
           were computed

    Output: phi, angles of the polar coordinates
            rho, radius of the polar coordinates with respect to the center
    """

    # get the abscissas and ordinates of the contour
    xcon = cnt[:, 0][:, 0]
    ycon = cnt[:, 0][:, 1]

    # translate to the center
    xs = xcon - center[0]
    ys = ycon - center[1]

    # radius of the translated points from the center
    rho = np.sqrt(xs**2 + ys**2)

    # angle of the translated points
    phi = np.arctan2(ys, xs)

    # numpy atan2 returns from -pi to pi
    # if phi < 0:
    #     phi = phi + 2 * np.pi
    # else:
    #     phi
    phi[phi < 0] = phi[phi < 0] + 2 * np.pi
    # with open("cnt.csv", 'w') as f:
    #     print("x,", "y", file=f)
    #     for x, y in zip(phi, rho):
    #         print(x, ",", y, file=f)

    return phi, rho


def polar2cart(phi, rho, center):
    """
    Convert from polar coordinates to cartesian coordinates. The coordinates
    of the center of rotation is from the origin of the cartesian systems.

    Input: phi, numpy array of angles of the polar coordinates, in radians
           rho, numpy array of the radii with respect to center of rotation
           center, center of rotation in the cartesian coordinate system

    Ouput: xs, numpy array of abscissas
           ys, numpy array of ordinates
    """
    # center of rotation
    cX, cY = center

    # conversion from polar to cartesian
    xs = np.cos(phi) * rho + cX
    ys = np.sin(phi) * rho + cY

    return xs, ys


def smooth_contour(cnt, center, npts, span, sigma, angle, filename):
    """
    Smooth the outline countour generated by openCV using cubic spline
    interpolation. The contour points in cartesian coordinate systems is
    converted to polar coordinate system with center of mass as the center of
    rotation. Smoothing is performed in the polar coordinate system and these
    smoothed polar coordinates are returned. Also save an image of the original
    2D image and the smoothed contour for later reference.

    Input: cnt, array of array of points [[x1, y1], [x2, y2],... [xi,yi]]
           center, center of rotation for convertion from cartesian to
           polar coordinates
           npts, the number of vertices to describe half (180 degrees) the
           curve when converting from cartesian to polar and then smoothing
           span, for loess smoothing
           sigma, for gaussian smoothing
           rotation, rotation angle when the starting point is on the corner
           filename, filename of the 2D image of the particle for saving the
           image with smoothed contour

    Output: phi_sm, the angle in polar coordinate system of the smoothed line
            rho_sm, the radius in polar coordinate system of the smoothed line
            xsm, the smoothed x-coordinates
            ysm, the smoothed y-coordinates
    """
    # create an empty image, openCV is (B, G, R)
    # colors from colorbrewer,
    # http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
    dark_orange = (253, 174, 97)
    light_yellow = (255, 255, 191)

    # if the starting point is right on one of the corners then it fails to
    # fit a circle at that corner, so the temporary work around right now
    # is to rotate the countours by a small angle and then segment the line
    # and fit the circles
    if angle:
        # extract the x and y coordinates
        x = cnt[:, 0][:, 0]
        y = cnt[:, 0][:, 1]
        cX = center[0]
        cY = center[1]
        xr = (x - cX) * np.cos(angle) + (y - cY) * np.sin(angle) + cX
        yr = -(x - cX) * np.sin(angle) + (y - cY) * np.cos(angle) + cY
        cnt = np.array([[[int(x), int(y)]] for x, y in zip(xr, yr)])

    # change from cartesian coordinate system to polar coordinate system
    # with respect to the centroid, the radii are the two distances to the
    # the edge of the particle from the centroid,
    # num_of_pts are the number of verticles along the contour
    phis, rhos = cart2polar(cnt, center)

    # sort the arrays to arrange the data in ascending order
    p = phis.argsort()
    phi = phis[p]
    rho = rhos[p]

    # append part of the array to the end for smooth transition from the
    # last point to the first point, does not seem to help, visit later
    # idx_sig = 5 * sigma
    # phi = np.append(phi, phi[1:idx_sig] + phi[-1], axis=0)
    # rho = np.append(rho, rho[1:idx_sig], axis=0)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(phi, rho)
    # plt.show()

    # spline interpolation and smoothing
    tck = spinter.splrep(phi, rho, k=3, s=3)
    phi_sm = np.arange(0, 2 * np.pi, np.pi/npts)
    rho_sm = spinter.splev(phi_sm, tck, der=0)

    # loess smoothing, uses R loess if rloess_smooth
    # rho_sm = lowess_ar.lowess(phi, rho, phi_sm, deg=2, l=span)
    rho_sm = rloess_smooth(phi_sm, rho_sm, phi_sm, deg=2, span=span)

    # too many smoothing functions, gaussian filter
    rho_sm = gaussian_filter1d(rho_sm, sigma)

    # smoothed contour in cartesian coordinates for plotting
    xsm, ysm = polar2cart(phi_sm, rho_sm, center)

    # close the contour
    xsm[-1] = xsm[0]
    ysm[-1] = ysm[0]

    # create array of points contour boundary, for drawing in openCV
    cnt_sm = np.array([[int(x), int(y)] for x, y in zip(xsm, ysm)])

    # create a blank image with white background, draw the image of
    # of the smoothed particle and draw the original particle shape too
    img = create_blank(640, 480, rgb_color=(255, 255, 255))
    img = cv2.drawContours(img, [cnt_sm], 0, light_yellow, 2)
    img = cv2.drawContours(img, [cnt], 0, dark_orange, 2)
    cv2.imwrite('../plots/' + filename + 'sm_cnt.png', img)

    # return the array of smoothed contour
    return phi_sm, rho_sm, xsm, ysm


def max_inscribed_circle(img, filename):
    """
    Determine the largest inscribed circle.

    Input: img, binary image

    Output: (idx_col, idx_row), inscribed_radius, the center of the maximum
            inscribed circle and its radius
    """

    # Calculates the distance to the closest zero pixel for each pixel,
    # the particle is white which is the non-zero (255), the background is
    # black black pixel have zero value. So distance transform is computing
    # the shorted distance of each pixel inside the particle to boundary of
    # the particle. DIST_L2 is Euclidean distance
    edt = cv2.distanceTransform(img, distanceType=cv2.DIST_L2,
                                maskSize=cv2.DIST_MASK_PRECISE)

    # the maximum pixel value, the farthest pixel value is the radius
    inscribed_radius = np.max(edt)

    # to determine the center, get the position of the max value, farthest
    # pixel, the way argmax works, it is necessary to unravel the index to
    # convert from 1D indexing to 2D indexing. The row and column indices
    # returned depends on the unravelling method. The default is (row, col),
    # therefore the center is (col, row)
    idx_max = np.argmax(edt)
    idx_row, idx_col = np.unravel_index(idx_max, img.shape)

    # save the figure
    fig, ax = plt.subplots()
    incircle = plt.Circle((idx_col, idx_row), inscribed_radius, fill=False)
    # cax = ax.imshow(edt, cmap=matplotlib.cm.gray)
    cax = ax.imshow(edt)
    fig.colorbar(cax)
    ax.plot(idx_col, idx_row, 'kx')
    ax.add_artist(incircle)
    plt.savefig("../plots/" + filename + "_maxinscir.png")
    plt.close()

    return {'maxInscribedCircle': ((idx_col, idx_row), inscribed_radius)}


def calc_sphericity(params):
    """
    Compute different sphericity values
    Input: params, dictionary with different shape parameters computed from
           from the smoothed contour

    Output: different sphericity values
    """
    # 2d project area of the particle
    area_2d = params['area']

    # area of the minimum subscribing circle
    rad_mec = params['minEnclosingCircle'][1]
    area_mec = np.pi * rad_mec**2

    # area sphericity
    sa = area_2d/area_mec

    # diameter of the circle having the same area as the projected area
    dia_2d = np.sqrt(4.0/np.pi * area_2d)

    # diameter sphericity
    sd = dia_2d/(2.0 * rad_mec)

    # circle ratio sphericity
    sc = 2 * params['maxInscribedCircle'][1]/(2.0 * rad_mec)

    # perimeter sphericity
    sp = np.pi * dia_2d/params['perimeter']

    # width to length ratio sphericity
    swl = min(params['minAreaRect'][1])/max(params['minAreaRect'][1])

    return {'sa': sa, 'sd': sd, 'sc': sc, 'sp': sp, 'swl': swl}


def maxlinedev(x, y):
    """
    Find the point of maximum deviation in the contour from a line joining
    two points on the contour.

    Input: x, numpy array of abscissas of the contour
           y, numpy array of ordinates of the contour

    Output: maxdev, maximum deviation
            idxmaxdev, index of the point of maximum deviation
    """
    # this algorithm is based on:
    # http://www.peterkovesi.com/matlabfns/LineSegments/maxlinedev.m
    # by Peter Kovesi

    # error checking and handling, include later
    if len(x) == 1:
        print("WARNING: Contour contains only one point")

    # Distance between the end points
    x2mx1 = x[-1] - x[0]
    y2my1 = y[-1] - y[0]
    dist = np.sqrt((x2mx1)**2 + (y2my1)**2)

    if dist > np.finfo(float).eps:
        # distance from the line to each point on the contour
        dev = np.abs(x2mx1 * y[0] - y2my1 * x[0] - x2mx1 * y + y2my1 * x)/dist
    else:
        # end points are coincident, calcuate distance from the first point
        dev = calc_dist(x[0], y[0], x, y)

    # maximum deviation and index of maximum deviation
    maxdev = np.max(dev)
    idxmaxdev = np.argmax(dev)

    return maxdev, idxmaxdev


def lineseg(x, y, tol):
    """
    Form straight line segments from an boundary list.

    Input: x, numpy array of abscissas of the boundary
           y, numpy array of ordinates of the boundary
           tol, maximum deviation from the straight line before a segment
           is broken in two (measured in pixels)

    Output: array of numpy array containing
    """
    # ref: This algorithm is based on Peter Kovesi's
    # http://www.peterkovesi.com/matlabfns/LineSegments/lineseg.m

    # indices of the first and last point of the contour
    fst = 0
    lst = len(x)

    # list to hold the list of segmented points
    seglist = [[x[0], y[0]]]

    while fst < lst:
        # find the size and position of the maximum deviation
        maxdev, idxmaxdev = maxlinedev(x[fst:lst], y[fst:lst])
        # print(maxdev, idxmaxdev)
        # while the maximum deviation from the line > tol shorten the line to
        # the point of maximum deviation by adjusting the last point
        while maxdev > tol:
            lst = fst + idxmaxdev + 1
            # print('lst: ', lst)
            maxdev, idxmaxdev = maxlinedev(x[fst:lst], y[fst:lst])

        # add the last point for which deviation is less than tol to the
        # segemented list
        seglist.append([x[lst-1], y[lst-1]])
        # print('seglist: ', seglist)

        # rest the first and the last point for the next iteration
        fst = lst
        lst = len(x)

    return np.asarray(seglist)


def linecoef(p1, p2):
    """
    Coefficients A, B, C of line equation (Ax + By = C) from
    the two endpoints p1, p2 of the line

    Input: p1, p2 the two points (x, y) of the line segment

    Output: Coefficients of the line equation
    """
    # ref: https://stackoverflow.com/questions/20677795/
    #      how-do-i-compute-the-intersection-point-of-two-lines-in-python

    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    """
    Find the intersection point of the two lines
    A1 * x + B1 * y = C1
    A2 * x + B2 * y = C2
    The intersection point is:
    x = Dx/D
    y = Dy/D
    The determinant D
    A1 B1 which is      L1[0] L1[1]
    A2 B2               L2[0] L2[1]
    Dx:
    C1 B1 which is      L1[2] L1[1]
    C2 B2               L2[2] L2[1]
    and Dy:
    A1 C1 which is      L1[0] L1[2]
    A2 C2               L2[0] L2[2]

    Input: The coefficients returns by linecoef for Line1, Line2
           the two intersecting lines

    Output: x, y the point of intersection
    """

    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def convexpoints(seglist, center):
    """
    Not true deterination of convex points. Determine whether a point is
    convex based on the immediate neighbouring points. If the intersection of a
    ray from the specified center with the line joining the immediate
    neighbouring points is closer to the center than the point itself then the
    point is not convex.

    Input: seglist, an array of points [[x1, y1], [x2, y2], ...]
           center, coordinates of the center point

    Ouput: convex, an array of convex points [[x1, y1], [x2, y2], ...]
           concave, an array of concave points [[x1, y1], [x2, y2], ...]
    """

    # center
    cX, cY = center

    # create an empty list hold the coordinate of convex points
    concave = []
    convex = []

    # save the length of the seglist
    seglen = len(seglist)
    # prepend the second last point before the first, this helps to check if
    # the first point is convex or concave, the last and the first points are
    # the same that is why we need to use the second last point.
    seglist = np.insert(seglist, 0, seglist[-2], axis=0)

    # append the second point to the end of the array, which is now the third
    # element after prepending the second last element before the first element
    # seglist = np.append(seglist, [seglist[2]], axis=0)

    # from the first to the second last point check for convex points
    for i in np.arange(0, seglen-1):
        # calculate distance between the center and the middle element
        cdist = np.sqrt((seglist[i+1][0] - cX)**2 + (seglist[i+1][1] - cY)**2)
        linea = linecoef([seglist[i][0], seglist[i][1]],
                         [seglist[i+2][0], seglist[i+2][1]])
        lineb = linecoef([seglist[i+1][0], seglist[i+1][1]], [cX, cY])

        intx, inty = intersection(linea, lineb)

        intdist = np.sqrt((intx - cX)**2 + (inty - cY)**2)

        if intdist:
            if intdist >= cdist:
                concave.append(seglist[i+1])
            else:
                convex.append(seglist[i+1])
        else:
            print('WARNING: No single intersection point detected')

    return np.asarray(convex), np.asarray(concave)


# these circle fitting technique has been copied from SciPy Cookbook
# http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
# the least squares with Jacobian is used because it was stated it was the
# fastest based on the number of calls
def fc(center, x, y):
    """ calculate the algebraic distance between the 2D points and
    the mean circle centered at c=(xc, yc) """
    xc, yc = center
    ri = calc_dist(xc, yc, x, y)
    return ri - np.mean(ri)


def Dfc(center, x, y):
    """ Jacobian of fcircle
    The axis corresponding to derivatives
    must be coherent with the col_deriv option of leastsq"""
    xc, yc = center
    dfc = np.empty((2, x.size))

    radius = calc_dist(xc, yc, x, y)
    # df/dxc and df/dyc
    dfc[0] = (xc - x)/radius
    dfc[1] = (yc - y)/radius
    dfc = dfc - dfc.mean(axis=1)[:, np.newaxis]

    return dfc


def circle_lstsq(x, y):
    xc0 = np.mean(x)
    yc0 = np.mean(y)
    center = (xc0, yc0)
    center_new, ier = leastsq(fc, center, args=(x, y),
                              Dfun=Dfc, col_deriv=True)

    xc_new, yc_new = center_new
    ri = calc_dist(xc_new, yc_new, x, y)
    radius = np.mean(ri)

    return center_new, radius


def f_3b(beta, x):
    """ implicit definition of the circle """
    return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 - beta[2]**2


def jacb(beta, x):
    """ Jacobian function with respect to the parameters beta.
    return df_3b/dbeta
    """
    xc, yc, r = beta
    xi, yi = x

    df_db = np.empty((beta.size, x.shape[1]))
    df_db[0] = 2*(xc-xi)                     # d_f/dxc
    df_db[1] = 2*(yc-yi)                     # d_f/dyc
    df_db[2] = -2*r                           # d_f/dr

    return df_db


def jacd(beta, x):
    """ Jacobian function with respect to the input x.
    return df_3b/dx
    """
    xc, yc, r = beta
    xi, yi = x

    df_dx = np.empty_like(x)
    df_dx[0] = 2*(xi-xc)                     # d_f/dxi
    df_dx[1] = 2*(yi-yc)                     # d_f/dyi

    return df_dx


def calc_estimate(data):
    """ Return a first estimation on the parameter from the data  """
    xc0, yc0 = data.x.mean(axis=1)
    r0 = np.sqrt((data.x[0]-xc0)**2 + (data.x[1]-yc0)**2).mean()
    return xc0, yc0, r0


def circle_3b(xs, ys):
    # for implicit function
    # data.x contains both coordinates of the points
    # data.y is the dimensionality of the response
    lsc_data = odr.Data(np.row_stack([xs, ys]), y=1)
    lsc_model = odr.Model(f_3b, implicit=True,
                          estimate=calc_estimate, fjacd=jacd, fjacb=jacb)
    # beta0 has been replaced by an estimate function
    lsc_odr = odr.ODR(lsc_data, lsc_model)
    # use user derivatives function without checking
    lsc_odr.set_job(deriv=3)
    # print details for each iteration
    # lsc_odr.set_iprint(iter=1, iter_step=1)
    lsc_out = lsc_odr.run()

    xc_3b, yc_3b, R_3b = lsc_out.beta

    return (xc_3b, yc_3b), R_3b


def corner_circles(convex, cnt, mic, factor, npts, pxltol, tol, maxiter):
    """
    Fit circles to the corners
    Input: convex, array of points[[x1, y1], [x2, y2], ...]
           identified as potential convex corner points
           cnt, [[]]
           cnt, array of points [[x1, y1], [x2, y2], ...]
    """

    # list of circle centers and radius
    cir_centers = []
    cir_radius = []

    # figure out what this range is for
    cir_range = []

    # indices of the first and last point
    fp = 0
    lp = len(convex)

    convex = np.append(convex, [[convex[0, 0], convex[0, 1]]], axis=0)

    # counter to prevent infinite looping, the maximum iteration for n points
    # will not be more than n * (n-1) in this case even if no circles are
    # found. Need to refine this with proper counting theorem but that is for
    # later
    nmax = lp * (lp - 1)
    n = 0
    while lp >= (fp + npts) and n < nmax:
        # print("fp, lp, fp+npts", fp, lp, fp+npts)
        # consider points from starting point (fp) and ending on the
        # last point to fit the circle
        # center_est = (np.mean(convex[fp:lp, 0]), np.mean(convex[fp:lp, 1]))
        center, radius = circle_lstsq(convex[fp:lp, 0], convex[fp:lp, 1])
        # center, radius, residual = gfc.fitcircle(
        #     convex[fp:lp, 0], convex[fp:lp, 1], tol, maxiter)
        # center, radius = circle_3b(convex[fp:lp, 0], convex[fp:lp, 1])
        # compute the radial distance from the new center to all the
        # boundary points of the smoothed contour
        min_dist = cv2.pointPolygonTest(cnt, (center[0], center[1]),
                                        measureDist=True)

        # check if the point lies within the particle,
        # revist this to see if it can be done better by removing the
        # nested if statements
        if min_dist < 0:
            # the point does not lie within the particle
            if lp > (fp + npts):
                # drop the last point and start over again
                lp = lp - 1
            else:
                # no circle found for the given point move to next point
                fp += 1
                lp = len(convex)
        elif min_dist >= 0:
            # the point lines within the particle
            if radius < factor * min_dist or\
             np.abs(radius - min_dist) <= pxltol and radius <= mic:
                # Yeah found a fitting circle
                cir_centers.append([center[0], center[1]])
                cir_radius.append(radius)
                cir_range.append([fp, lp])
                fp = lp + 1
                lp = len(convex)
            elif lp > (fp + npts):
                # if circle not found drop the last point and start over again
                    lp = lp - 1
            else:
                # if circle not found for the given point
                # move to the next point
                fp += 1
                # if no circle found
                lp = len(convex)
                # print('Point inside Here 2')
        n += 1
    return cir_centers, cir_radius, cir_range


if __name__ == "__main__":
    # path for input and output files, change this path accordingly
    infilepath = "../data/"
    outfilepath = "../plots/"

    # filename for the summary file
    sumfilename = "py_summary.txt"

    # scale of the image, if 247 pixels = 1 mm
    # scale = 1/247 mm/pixel
    scale = 1.0

    # Tolerance and maximum iteration for the Gauss-Newton method
    TOL = 1e-5
    MAXITER = 1000

    # maxium deviation during line segmenting
    MAXDEV = 0.2

    # Tolerance for radius of the fitted circle
    RADTOL = 1.02

    # Tolerance for radius of the fitted circle in terms of pixel
    PXLTOL = 2

    # Minimum number of points for circle fitting
    circle_minpts = 3

    # number of points describing the contour in the spline smoothing
    spline_npts = 2048

    # span in loess
    loess_span = 0.07

    # rotation in radians
    rot = np.pi/180 * 3

    # sigma used in gaussian smoothing, 12?
    gauss_sigma = 5

    # params = {'center_of_mass': (0, 0), 'area': 0, 'perimeter': 0,
    #           'minEnclosingCircle': ((0, 0), 0), 'aspect_ratio': 0,
    #           'diameter': 0, 'MajMinAxis': (0, 0), 'solidity': 0,
    #           'incircle_center': (0, 0), 'incircle_radius': 0}
    params = {}

    # read and store in array all the files in the path ending with
    images = [f for f in os.listdir(infilepath) if f.endswith('.png')]

    # sort the files according to the numeral in the file name
    images = natsort.natsorted(images)

    # print the names of the sorted image names
    print('Sorted filenames: ', images)

    # warning messages
    warningcontour = "WARNING: More than one countour found"

    # string format for heading
    hdfmtstr = "{0:20s},{1:5s},{2:5s},{3:5s},{4:5s},{5:5s},{6:5s}"
    valfmtstr = "{0:20s},{1:5.3f},{2:5.3f},{3:5.3f},{4:5.3f},{5:5.3f},\
    {6:5.2f}"

    # open summary file for writing the results of all the analysis
    f = open(outfilepath + sumfilename, 'w')

    # write the heading
    print(hdfmtstr.format("Filename", "Sa", "Sd", "Sc", "Sp", "Swl", "Round"),
          file=f)

    for image in images:
        # file name without the extension, this will be used to
        # generate the output filename
        imagenoext = os.path.splitext(image)[0]

        # read image file
        img = cv2.imread(infilepath + image, cv2.IMREAD_GRAYSCALE)

        # apply Gaussian filter
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)

        # threshold
        # set values equal to or above 127 to 0
        # set values below 220 to 255
        # ret, img_th = cv2.threshold(img_blr, 127, 255, cv2.THRESH_BINARY_INV)
        # Otsu's thresholding after Gaussian filtering
        ret, img_th = cv2.threshold(img_blur, 0, 255,
                                    cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # fill holes inside the particles
        img_fillholes = fillholes(img_th)

        # create the contour around the edge of the particle
        im2, cnts, heir = cv2.findContours(img_fillholes,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

        # there will be a single particle in an image, therefore
        # there should be only one contour. If more than one contour is
        # encountered then display a worning and log it
        if len(cnts) > 1:
            warning = warningcontour
            print(warningcontour)
        else:
            warning = ""

        # if multiple contours were found then choose the one wth the
        # largest area,
        # confirm that the list of point starts at the 90 deg, N or y-axis and
        # moves counter-clockwise
        cnt = max(cnts, key=cv2.contourArea)

        # find the center of mass from the moments of the contour
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # smoothed points in polar and cartesian system
        phi_sm, rho_sm, xsm, ysm = smooth_contour(
            cnt, (cX, cY), spline_npts, loess_span,
            gauss_sigma, rot, imagenoext)

        # create a blank binary image
        img_sm = np.zeros((480, 640), dtype=np.uint8)
        # list of array of [x, y] points for the smoothed contour
        cnt_sm = np.array([[int(x), int(y)] for x, y in zip(xsm, ysm)])
        # update particle parameters from the outer smoothed contour
        params.update(particle_countour_params(cnt_sm))

        # take the smoothed contour and then create the smoothed particle
        # in white
        img_sm = cv2.fillPoly(img_sm, [cnt_sm], 225)

        # update the parameters with the center and radius of the maximum
        # inscribed circle (center(x, y), radius)
        params.update(max_inscribed_circle(img_sm, imagenoext))

        # update the param with sphericity values
        params.update(calc_sphericity(params))

        # straight line segments of specified deviation fitted to list of
        # boundary points
        seglist = lineseg(xsm, ysm, MAXDEV)

        # list of convex points
        # a convex point is defined as the point which is closer to the
        # centroid than the intersection point of:
        # (a) line draw with the point's neighbours as the end points
        # (b) a ray drawn from the centroid in the direction of the point
        pconvex, pconcave = convexpoints(seglist, params['center_of_mass'])

        # mic is the maximum inscribed circle radius
        mic = params['maxInscribedCircle'][1]
        cc, cr, crg = corner_circles(pconvex, cnt_sm, mic, RADTOL,
                                     circle_minpts, PXLTOL, TOL, MAXITER)

        # compute Wadell's roundness as the ratio of the average radius of the
        # fitted circles and the radius of the largest inscribed circle
        roundness = np.mean(cr)/mic

        # write the values to the file
        print(valfmtstr.format(imagenoext, params['sa'], params['sd'],
              params['sc'], params['sp'], params['swl'], roundness), file=f)

        # for plotting
        xori = cnt[:, 0][:, 0]
        yori = cnt[:, 0][:, 1]
        xseg = seglist[:, 0]
        yseg = seglist[:, 1]

        # add the maximum inscribed circle to the list
        cc.append(params['maxInscribedCircle'][0])
        cr.append(mic)
        # to draw the circles
        patches = [plt.Circle(ci, ri, fill=False) for ci, ri in zip(cc, cr)]
        fig, ax = plt.subplots()
        ax.plot(cX, cY, 'x', markersize=10)
        ax.plot(xori, yori, '-')
        coll = matplotlib.collections.PatchCollection(patches,
                                                      match_original=True)
        ax.add_collection(coll)
        ax.plot(xseg, yseg, 's-', mfc='None')
        ax.plot(pconvex[:, 0], pconvex[:, 1], 'x')
        ax.text(pconvex[0, 0], pconvex[0, 1], "0")
        ax.text(pconvex[1, 0], pconvex[1, 1], '1')
        ax.text(pconvex[-1, 0], pconvex[-1, 1], str(len(pconvex)))
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        # circle = plt.Circle(center_new, rnew, fill=False)
        # ax.add_artist(circle)
        # ax.set_xlim(150, 500)
        # ax.set_ylim(90, 390)
        ax.set_aspect('equal')
        plt.show()
    f.close()
