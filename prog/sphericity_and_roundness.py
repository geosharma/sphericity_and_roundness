# ! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Description: Compute sphericity and roundness of a particle.
Create binary images of particles with sufficient resolution, one per image
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
import matplotlib
import matplotlib.pyplot as plt

# import module for image manipulation
import cv2

# import files with all the helper function from sph_rnd_func.py (srf)
import sph_rnd_func as srf


# path for input and output files, change this path accordingly
infilepath = "../data/"
outfilepath = "../plots/"

# filename for the summary file
sumfilename = "summary.txt"

# scale of the image, if 247 pixels = 1 mm
# scale = 1/247 mm/pixel
scale = 1.0

# maximum deviation during line segmentation
dev = 0.3

# data span for loess
span = 0.07

# tolerance for fitted circle radius when compared to the shortest distance
# to the boundary from the center
radfac = 1.02

# the smallest number of points to which a circle can be fitted
minpts = 3

# rotate the image, if a corner is close to the starting and ending points
rot = 2

# string format for heading
hdfmtstr = "{0:20s},{1:7s},{2:5s},{3:5s},{4:5s},{5:5s},{6:5s},{7:5s}"
valfmtstr = "{0:20s},{1:7.2f},{2:5.3f},{3:5.3f},{4:5.3f},{5:5.3f},{6:5.3f},\
{7:5.3f}"

# params = {'centroid': (0, 0), 'area': 0, 'perimeter': 0,
#           'minEnclosingCircle': ((0, 0), 0), 'aspect_ratio': 0,
#           'diameter': 0, 'ellipse': 0, 'MajMinAxis': (0, 0), 'solidity': 0,
#           'maxInscribedCircle': ((idx_col, idx_row))}
params = {}

# read and store in array all the files in the path ending with
images = [f for f in os.listdir(infilepath) if f.endswith('.png')]

# sort the files according to the numeral in the file name
images = natsort.natsorted(images)

# print the names of the sorted image names
print('Sorted filenames: ', images)

# warning messages
warningcontour = "WARNING: More than one countour found"

# open summary file for writing the results of all the analysis
f = open(outfilepath + sumfilename, 'w')

# write the heading
print(hdfmtstr.format("Filename", "Dia.", "Sa", "Sd", "Sc", "Sp", "Swl",
                      "Rnd"), file=f)

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
    img_fillholes = srf.fillholes(img_th)

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
    # cnt = cv2.approxPolyDP(contour, 0.5, True)

    # find the center of mass from the moments of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # smoothed points in polar and cartesian system
    phi_sm, rho_sm, xsm, ysm = srf.smooth_contour(cnt, (cX, cY),
                                                  span, imagenoext)

    # create a blank binary image
    img_sm = np.zeros((480, 640), dtype=np.uint8)

    # list of array of [x, y] points for the smoothed contour
    cnt_sm = np.array([[int(x), int(y)] for x, y in zip(xsm, ysm)])

    # update particle parameters from the outer smoothed contour
    params.update(srf.particle_countour_params(cnt_sm))

    # take the smoothed contour and then create the smoothed particle
    # in white
    img_sm = cv2.fillPoly(img_sm, [cnt_sm], 225)

    # update the parameters with the center and radius of the maximum
    # inscribed circle (center(x, y), radius)
    params.update(srf.max_inscribed_circle(img_sm, imagenoext))

    # update the param with sphericity values
    params.update(srf.calc_sphericity(params))

    xori = cnt[:, 0][:, 0]
    yori = cnt[:, 0][:, 1]

    seglist = srf.lineseg(xsm, ysm, dev)
    xseg = seglist[:, 0]
    yseg = seglist[:, 1]

    # determine the convex and concave points
    pconvex, pconcave = srf.convexpoints(seglist, params['centroid'])

    # fit circles to the particle
    mic = params['maxInscribedCircle'][1]
    cc, cr, crg = srf.corner_circles(pconvex, cnt_sm, mic, radfac, minpts)
    # compute roundness
    roundness = 0.0
    for i, ri in enumerate(cr):
        roundness += ri
    roundness = (roundness/(i + 1))/mic
    print(np.mean(cr)/mic)
    print('roundness: ', roundness)
    print('mic: ', mic)
    print(valfmtstr.format(imagenoext, params['diameter'], params['sa'],
                           params['sd'], params['sc'], params['sp'],
                           params['swl'], roundness), file=f)
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
    plt.savefig("../plots/" + imagenoext + "_cir.png", ext="png")
f.close()
