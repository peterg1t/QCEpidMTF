



###########################################################################################
#
#   Script name: qc-epidmtf
#
#   Description: Tool for calculating epid modlation transfer function (MTF) using a QCMV phantom (Standard Imaging).
#
#   Example usage: python qc-epidmtf "/file/"
#
#   Author: Pedro Martinez
#   pedro.enrique.83@gmail.com
#   5877000722
#   Date:2019-04-09
#
###########################################################################################

import os
import sys

# sys.path.append('C:\Program Files\GDCM 2.8\lib')
import pydicom
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import numpy as np
import argparse
import cv2
from skimage.feature import blob_log
from math import *



def running_mean(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1

        # cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


# axial visualization and scrolling
def viewer(volume, dx, dy,center):
    # remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    extent = (0, 0 + (volume.shape[1] * dx),
              0, 0 + (volume.shape[0] * dy))
    # img=ax.imshow(volume, extent=extent)
    img=ax.imshow(volume)
    # ax.set_xlabel('x distance [mm]')
    # ax.set_ylabel('y distance [mm]')
    ax.set_xlabel('x pixel')
    ax.set_ylabel('y pixel')
    # ax.set_title("Phantom image=")
    fig.suptitle('Image', fontsize=16)
    fig.colorbar(img, ax=ax, orientation='vertical')
    # fig.canvas.mpl_connect('key_press_event', process_key_axial)
    for x,y in center:
        ax.scatter(x,y)





def shape_detect(c):
    shape='unidentified'
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c) #number of vertices in the contour
    if len(approx)==3:
        shape='triangle'
    elif len(approx)==4:
        #compute the bounding box of the contour and find the aspect ratio
        (x,y,w,h)=cv2.boundingRect(approx)
        ar=w/float(h)

        shape='square' if ar >=0.95 and ar <=1.05 else 'rectangle'
    else:
        shape='circle'

    return shape







def read_dicom(filename,ioption):
    if os.path.splitext(filename)[1] == '.dcm':
        dataset = pydicom.dcmread(filename)
        SID = dataset.RTImageSID
        dx = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[0]) / 1000)
        dy = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[1]) / 1000)
        print("pixel spacing row [mm]=", dx)
        print("pixel spacing col [mm]=", dy)
        ArrayDicom = dataset.pixel_array

        if ioption.startswith(('y', 'yeah', 'yes')):
            max_val = np.amax(ArrayDicom)
            ArrayDicom = ArrayDicom / max_val
            min_val = np.amin(ArrayDicom)
            ArrayDicom = ArrayDicom - min_val
            ArrayDicom = (1 - ArrayDicom)  # inverting the range

            min_val = np.amin(ArrayDicom)  # normalizing
            ArrayDicom = ArrayDicom - min_val
            ArrayDicom = ArrayDicom / (np.amax(ArrayDicom))
        else:
          min_val = np.amin(ArrayDicom)
          ArrayDicom = ArrayDicom - min_val
          ArrayDicom = ArrayDicom / (np.amax(ArrayDicom))

        ArrayDicom= 255* ArrayDicom
        # print(ArrayDicom.dtype)

        ArrayDicom=ArrayDicom.astype(np.uint8)




#-----------------------------------------------------------------------------------------------
        #performing bilateral filtering to remove some noise without affecting the edges
        img_bifilt=cv2.bilateralFilter(ArrayDicom,11,17,17)
        edged=cv2.Canny(img_bifilt,30,200)

        # #thresholding the image to find the square
        # th=cv2.threshold(img_bifilt,120,255,cv2.THRESH_TRUNC)[1]
        # th2=cv2.adaptiveThreshold(th,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # th2=cv2.bitwise_not(th2)
        # # edges=cv2.Canny(th2,60,120)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # img=cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

        contours=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
        for cnt in contours:
            print(cv2.contourArea(cnt))
        # print('contours_detected=',len(contours))



        cv2.drawContours(edged,contours, 0, (10,0,160), 3)
        # cv2.imshow('img',th2)
        # cv2.waitKey(0)
        # # cv2.imshow('img',img_bifilt)
        # # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # th=cv2.adaptiveThreshold(ArrayDicom,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        plt.figure()
        plt.imshow(ArrayDicom)
        plt.title('original')

        plt.figure()
        plt.imshow(edged)
        plt.title('bilinear filtered')
        #
        # plt.figure()
        # plt.imshow(th)
        # plt.title('threshold applied')
        #
        # plt.figure()
        # plt.imshow(th2)
        # plt.title('2 threshold')
        #
        # # plt.figure()
        # # plt.imshow(edges)
        # # plt.title('edges detected')
        #
        # # plt.figure()
        # # plt.imshow(img)
        # # plt.title('morphology applied (dil&ero)')

        plt.show()



        exit(0)
# -----------------------------------------------------------------------------------------------





        #lets fin the circles in this QC3 phantom (Only for Manitoba QC3 the Standard imaging one does not have them)

        circles=cv2.HoughCircles(ArrayDicom,cv2.HOUGH_GRADIENT,2,50,param1=100,param2=40,minRadius=2,maxRadius=15)
        circles=np.uint16(np.around(circles))


        centerXRegion = []
        centerYRegion = []
        center=[]
        centerRRegion = []
        grey_ampRegion = []


        for i in range(0,2):
            print(circles[:,i,0],circles[:,i,1],circles[:,i,2])
            center.append((circles[:,i,0],circles[:,i,1]))

        viewer(ArrayDicom, dx, dy, center)

    #Now that we have correctly detected the points we need to estimate the scaling of the image and the location of every ROI

    x1,y1=center[0]
    x2,y2=center[1]

    theta=atan(abs(y2-y1)/abs(x2-x1))
    theta_deg=degrees(theta)

    print('theta_deg (angle the phantom was placed) =',theta_deg)

    #The ROIs location can be identified by its positions with respect to the two points


    # #Let's create a mask
    # mask=np.zeros(np.shape(ArrayDicom),dtype=np.uint8)
    # roi1_corners=np.array([[ (525,330),(571,375),(506,442) ,(458,395) ]],dtype=np.int32)
    #
    # cv2.fillPoly(mask,roi1_corners,255)
    # result=cv2.bitwise_and(ArrayDicom,mask)
    #
    # plt.figure()
    # plt.imshow(result)
    # plt.title('mask')

    # mser = cv2.MSER_create()
    # regions = mser.detectRegions(ArrayDicom,None)[0]
    # print('number of regions=',np.shape(regions))
    # # hulls=[cv2.convexHull(p.reshape(-1,1,2)) for p in regions]
    # hulls=cv2.convexHull(regions)
    #
    # cv2.polylines(ArrayDicom,hulls,1,(10,255,0))
    #
    # plt.figure()
    # plt.imshow(ArrayDicom)
    # plt.title('original')
    # plt.show()




    # Normal mode:
    print()
    print("Filename.........:", filename)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", dataset.PatientID)
    print("Modality.........:", dataset.Modality)
    print("Study Date.......:", dataset.StudyDate)
    print("Gantry angle......", dataset.GantryAngle)
    #
    # if 'PixelData' in dataset:
    #     rows = int(dataset.Rows)
    #     cols = int(dataset.Columns)
    #     print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
    #         rows=rows, cols=cols, size=len(dataset.PixelData)))
    #     if 'PixelSpacing' in dataset:
    #         print("Pixel spacing....:", dataset.PixelSpacing)
    #
    # # use .get() if not sure the item exists, and want a default value if missing
    # print("Slice location...:", dataset.get('SliceLocation', "(missing)"))
    plt.show()



parser = argparse.ArgumentParser()
parser.add_argument('epid',type=str,help="Input the filename")
args=parser.parse_args()
filename=args.epid

while True:  # example of infinite loops using try and except to catch only numbers
    line = input('Are these files from a clinac [yes(y)/no(n)]> ')
    try:
        ##        if line == 'done':
        ##            break
        ioption = str(line.lower())
        if ioption.startswith(('y', 'yeah', 'yes', 'n', 'no', 'nope')):
            break

    except:
        print('Please enter a valid option:')



read_dicom(filename,ioption)
