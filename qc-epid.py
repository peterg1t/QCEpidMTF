



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
from operator import itemgetter



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
    # ax.set_ylabel('y disfilename1=args.epidtance [mm]')
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






def mtf_calc(ROI, ROInoise):
    print('calculating MTF')
    # see doselab manual for method of calculation
    M5num=np.percentile(ROI[len(ROI)-1],90)-np.percentile(ROI[len(ROI)-1],10)
    M5den=np.percentile(ROI[len(ROI)-1],90)+np.percentile(ROI[len(ROI)-1],10)
    M5=M5num/M5den


    LinePairs = [0.76, 0.43, 0.23, 0.20, 0.1]
    MTF=[]
    for region in ROI:
        num=np.percentile(region,90)-np.percentile(region,10)
        den=np.percentile(region,90)+np.percentile(region,10)
        Mi=num/den
        MTF.append(Mi/M5)

    print(MTF)


    plt.figure()
    plt.plot(LinePairs,MTF)
    plt.title('rMTF plot')
    plt.xlabel('Line pairs lp/mm')
    plt.ylabel('MTF')
    plt.ylim((0,1))
    plt.xlim((0.1,0.76))
    plt.show(block=False)



def cnr_calc(ROI,ROInoise):
    #nothing here yet
    print('calculating CNR')
    plt.figure()
    plt.imshow(ROI[0])
    plt.title('ROI_0')
    plt.figure()
    plt.imshow(ROI[1])
    plt.title('ROI_1')
    plt.figure()
    plt.imshow(ROInoise[0])
    plt.title('ROInoise_0')
    plt.figure()
    plt.imshow(ROInoise[1])
    plt.title('ROInoise_1')



    mean_0=np.mean(ROI[0])
    mean_1=np.mean(ROI[1])

    contrast = 100 * abs(mean_0-mean_1)/(mean_0+mean_1)
    print('contrast=', contrast)

    # std_dev_noise_0=np.std(ROI[0])
    std_dev_noise_0=np.std(ROInoise[0])
    std_dev_noise_1=np.std(ROInoise[1])
    print('std_dev_noise_0=',std_dev_noise_0)

    noise= 100*sqrt(std_dev_noise_0*std_dev_noise_0+std_dev_noise_1*std_dev_noise_1)/sqrt(mean_0*mean_0+mean_1+mean_1)
    print('noise=',noise)


    cnr = contrast/noise
    # cnr = abs(mean_0-mean_1)/std_dev_noise_0


    print('cnr=',cnr)






    plt.show()
    exit(0)


























def read_dicom(filename1,filename2,ioption):
    if os.path.splitext(filename1)[1] == '.dcm':
        dataset = pydicom.dcmread(filename1)
        dataset2 = pydicom.dcmread(filename2)
        SID = dataset.RTImageSID
        dx = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[0]) / 1000)
        dy = 1 / (SID * (1 / dataset.ImagePlanePixelSpacing[1]) / 1000)
        print("pixel spacing row [mm]=", dx)
        print("pixel spacing col [mm]=", dy)

        ArrayDicom_o = dataset.pixel_array
        ArrayDicom = dataset.pixel_array
        ArrayDicom2 = dataset2.pixel_array


        # test to make sure image is displayed correctly bibs are high amplitude against dark background
        ctr_pixel = ArrayDicom[np.shape(ArrayDicom)[0] // 2, np.shape(ArrayDicom)[1] // 2]
        corner_pixel = ArrayDicom[0, 0]



        if ctr_pixel < corner_pixel:     #we need to invert the image range for both clinacs and tb
            max_val = np.amax(ArrayDicom)
            volume = ArrayDicom / max_val
            min_val = np.amin(volume)
            volume = volume - min_val
            volume = (1 - volume)  # inverting the range
            ArrayDicom = volume * max_val


            # max_val = np.amax(im_profile)
            # volume = ArrayDicom / max_val
            # min_val = np.amin(im_profile)
            # volume = volume - min_val
            # volume = (1 - volume)  # inverting the range







        rand_noise = ArrayDicom - ArrayDicom2  # we need the random noise so we can calculate the MTF function
        ArrayDicom_f= cv2.bilateralFilter(np.asarray(ArrayDicom,dtype='float32'), 33, 41, 17) #aggresive
        # ArrayDicom_f= cv2.bilateralFilter(np.asarray(ArrayDicom,dtype='float32'), 3, 17, 17) #mild
        # rand_noise_v2 =  np.asarray(ArrayDicom,dtype='float32') - ArrayDicom_f #noise removed from the bilateral filter
        rand_noise_v2 = ArrayDicom_f - np.asarray(ArrayDicom,dtype='float32')


        min_val = np.amin(ArrayDicom)  # normalizing
        ArrayDicom = ArrayDicom - min_val
        ArrayDicom = ArrayDicom / (np.amax(ArrayDicom)) #normalizing the data

        min_val = np.amin(ArrayDicom2)  # normalizing
        ArrayDicom2 = ArrayDicom2 - min_val
        ArrayDicom2 = ArrayDicom2 / (np.amax(ArrayDicom2)) #normalizing the data




        ArrayDicom= 255* ArrayDicom
        ArrayDicom2= 255* ArrayDicom2
        # print(ArrayDicom.dtype)

        ArrayDicom=ArrayDicom.astype(np.uint8)
        ArrayDicom2=ArrayDicom2.astype(np.uint8)



        #if we want the random noise of the image values (not the original values)
        #rand_nois, ROInoisee = ArrayDicom - ArrayDicom2  # we need the random noise so we can calculate the MTF function




#-----------------------------------------------------------------------------------------------
        #performing bilateral filtering to remove some noise without affecting the edges
        img_bifilt=cv2.bilateralFilter(ArrayDicom,11,17,17)
        # edged=cv2.Canny(img_bifilt,30,200)


        # #thresholding the image to find the square
        th=cv2.threshold(img_bifilt,200,255,cv2.THRESH_TRUNC)[1]
        th2=cv2.adaptiveThreshold(th,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        th2=cv2.bitwise_not(th2)





        # doing blob detection
        blobs_log = blob_log(th2, min_sigma=3, max_sigma=5, num_sigma=20, threshold=0.5,exclude_border=True)

        center=[]
        point_det=[]
        for blob in blobs_log:
            y, x, r = blob
            point_det.append((x,y,r))

        point_det=sorted(point_det,key=itemgetter(2),reverse=True)

        for i in range(0,2):
            x, y, r = point_det[i]
            center.append((int(round(x)),int(round(y))))


        viewer(ArrayDicom, dx, dy, center)









#         # # edges=cv2.Canny(th2,60,120)
#
#         # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
#         # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         # img=cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
#
#         # contours=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
#         # for cnt in contours:
#         #     print(cv2.contourArea(cnt))
#         # print('contours_detected=',len(contours))
#
#
#
#         # cv2.drawContours(edged,contours, 1, (10,0,160), 3)
#         # cv2.imshow('img',th2)
#         # cv2.waitKey(0)
#         # # cv2.imshow('img',img_bifilt)
#         # # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#
#
#         # th=cv2.adaptiveThreshold(ArrayDicom,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#         # plt.figure()
#         # plt.imshow(ArrayDicom)
#         # plt.title('original')
#         #
#         # plt.figure()
#         # plt.imshow(img_bifilt)
#         # plt.title('bilinear filtered')
#         #
#         # plt.figure()
#         # plt.imshow(th)blob
#         # plt.title('threshold applied')
#         #
#         plt.figure()
#         plt.imshow(th2)
#         plt.title('2 threshold')
#         #
#         # # plt.figure()
#         # # plt.imshow(edges)
#         # # plt.title('edges detected')
#         #
#         # # plt.figure()
#         # # plt.imshow(img)
#         # # plt.title('morphology applied (dil&ero)')
#
#         plt.show()
#
#
#--------------------------------------------------------------------






    #Now that we have correctly detected the points we need to estimate the scaling of the image and the location of every ROI
    x1,y1=center[0]
    x2,y2=center[1]

    theta=atan(abs(y2-y1)/abs(x2-x1))
    theta_deg=degrees(theta)

    print('theta_deg (angle the phantom was placed) =',theta_deg)

    #The distance between the centers of the ROIs in pixels is given by
    dist_horz_roi=int(21/dx)       #where 21 mm is the witdh of the ROI each ROI is 20 mm width with 1mm spacer and 28mm in height
    dist_vert_roi=int(28/dy)       #where 21 mm is the witdh of the ROI each ROI is 20 mm width with 1mm spacer and 28mm in height
    width_roi=int(20/dx)-10 # just subtracting a few pixels to avoid edge effects
    height_roi=int(27/dy)-10
    print('dist_horz_roi=',dist_horz_roi)


    #The ROIs location can be identified by its positions with respect to the two points
    #let's rotate the image around the center of the first ROI
    xrot=int(abs(x2+x1)/2)
    yrot=int(abs(y2+y1)/2)


    M = cv2.getRotationMatrix2D((xrot,yrot),theta_deg,1)
    ArrayDicom_rot=cv2.warpAffine(ArrayDicom_f,M,(np.shape(ArrayDicom_o)[1],np.shape(ArrayDicom_o)[0])) #if we want to use the real values
    # ArrayDicom_rot=cv2.warpAffine(ArrayDicom,M,(np.shape(ArrayDicom_o)[1],np.shape(ArrayDicom_o)[0]))
    rand_noise_rot=cv2.warpAffine(rand_noise,M,(np.shape(rand_noise)[1],np.shape(rand_noise)[0]))
    rand_noise_v2_rot=cv2.warpAffine(rand_noise_v2,M,(np.shape(rand_noise)[1],np.shape(rand_noise)[0]))

    plt.figure()
    plt.imshow(ArrayDicom_rot)
    plt.title('rotated')
    plt.show(block=False)

    ROImtf=[]
    ROImtf.append(ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot-int(width_roi/2):xrot+int(width_roi/2)])
    ROImtf.append(ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot-dist_horz_roi-int(width_roi/2):xrot-dist_horz_roi+int(width_roi/2)])
    ROImtf.append(ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot+dist_horz_roi-int(width_roi/2):xrot+dist_horz_roi+int(width_roi/2)])
    ROImtf.append(ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot-2*dist_horz_roi-int(width_roi/2):xrot-2*dist_horz_roi+int(width_roi/2)])
    ROImtf.append(ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot+2*dist_horz_roi-int(width_roi/2):xrot+2*dist_horz_roi+int(width_roi/2)])

    ROInoise = []
    ROInoise.append(rand_noise_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
               xrot - int(width_roi / 2):xrot + int(width_roi / 2)])
    ROInoise.append(rand_noise_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
               xrot - dist_horz_roi - int(width_roi / 2):xrot - dist_horz_roi + int(width_roi / 2)])
    ROInoise.append(rand_noise_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
               xrot + dist_horz_roi - int(width_roi / 2):xrot + dist_horz_roi + int(width_roi / 2)])
    ROInoise.append(rand_noise_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
               xrot - 2 * dist_horz_roi - int(width_roi / 2):xrot - 2 * dist_horz_roi + int(width_roi / 2)])
    ROInoise.append(rand_noise_rot[yrot - int(height_roi / 2):yrot + int(height_roi / 2),
               xrot + 2 * dist_horz_roi - int(width_roi / 2):xrot + 2 * dist_horz_roi + int(width_roi / 2)])



    ROIcnr = []
    ROIcnr.append(ArrayDicom_rot[yrot - dist_vert_roi - int(height_roi / 2):yrot - dist_vert_roi + int(height_roi / 2),
                  xrot - int(width_roi / 2):xrot + int(width_roi / 2)])
    ROIcnr.append(ArrayDicom_rot[yrot + dist_vert_roi - int(height_roi / 2):yrot + dist_vert_roi + int(height_roi / 2),
                  xrot - int(width_roi / 2):xrot + int(width_roi / 2)])


    ROIcnr_noise = []
    ROIcnr_noise.append(rand_noise_v2_rot[yrot - dist_vert_roi - int(height_roi / 2):yrot - dist_vert_roi + int(height_roi / 2),
                  xrot - int(width_roi / 2):xrot + int(width_roi / 2)])
    ROIcnr_noise.append(rand_noise_v2_rot[yrot + dist_vert_roi - int(height_roi / 2):yrot + dist_vert_roi + int(height_roi / 2),
                  xrot - int(width_roi / 2):xrot + int(width_roi / 2)])




    ROI1=ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot-int(width_roi/2):xrot+int(width_roi/2)]
    ROI2=ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot-dist_horz_roi-int(width_roi/2):xrot-dist_horz_roi+int(width_roi/2)]
    ROI3=ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot+dist_horz_roi-int(width_roi/2):xrot+dist_horz_roi+int(width_roi/2)]
    ROI4=ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot-2*dist_horz_roi-int(width_roi/2):xrot-2*dist_horz_roi+int(width_roi/2)]
    ROI5=ArrayDicom_rot[yrot-int(height_roi/2):yrot+int(height_roi/2),xrot+2*dist_horz_roi-int(width_roi/2):xrot+2*dist_horz_roi+int(width_roi/2)]


    # plt.figure()
    # plt.imshow(ROInoise[0])
    # plt.title('ROI1')
    #
    # plt.figure()
    # plt.imshow(ROInoise[1])
    # plt.title('ROI2')
    #
    # plt.figure()
    # plt.imshow(ROInoise[2])
    # plt.title('ROI3')
    #
    # plt.figure()
    # plt.imshow(ROInoise[3])
    # plt.title('ROI4')
    #
    # plt.figure()
    # plt.imshow(ROInoise[4])
    # plt.title('ROI5')
    #
    # plt.show()

    #now that we have the ROIs we can proceed to calculate the rMTF
    mtf_calc(ROImtf,ROInoise)

    #now that we have the ROIs we can proceed to calculate the CNR (contrast to noise ratio and the random noise)
    cnr_calc(ROIcnr,ROIcnr_noise)







    # Normal mode:
    print()
    print("Filename.........:", filename1)
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
parser.add_argument('epid1',type=str,help="Input the filename")
parser.add_argument('epid2',type=str,help="Input the filename")
# parser.add_argument('-a', '--add', nargs='?', type=argparse.FileType('r'), help='additional file for averaging before processing')
args=parser.parse_args()

filename1=args.epid1
filename2=args.epid2


# filename2=''
# if args.add:
#     additional=args.add
#     filename2=additional.name





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




read_dicom(filename1,filename2,ioption)
