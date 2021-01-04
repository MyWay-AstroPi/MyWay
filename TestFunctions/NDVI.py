import cv2
import numpy


DIFF_THRESHOLD = 0.3
PIXEL_THRESHOLD = 0.6


'''
UNKNOWN FUNCTION
'''
def contrastStretch(im):
    in_min = numpy.percentile(im, 5)
    in_max = numpy.percentile(im, 100)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out


def calculateNDVI(image):
    b, _, r = cv2.split(image) #Extraction of bgr valuer (ignore the g value)
    bottom = float(r) + float(b)
    bottom[bottom == 0] = 0.0000000000001 #Change 0s in bottom array (exclude the division by 0)
    ndvi = (float(r) - b) / bottom #Calculate NDVI value of each pixel

    ndvi = contrastStretch(ndvi) #???
    ndvi = ndvi.astype(numpy.uint8) #Conversion to numpy array

    return ndvi


def calculate_statistics(ndviArray, pixelThreshold = PIXEL_THRESHOLD, diffThreshold = DIFF_THRESHOLD):
    #Init table
    NDVIGrad = {
        0      : 0, #<0.1
        1      : 0, #0.1-0.2
        2      : 0, #0.2-0.3
        3      : 0, #0.3-0.4
        4      : 0, #0.4-0.5
        5      : 0, #0.5-0.6
        6      : 0, #0.6-0.7
        7      : 0, #0.7-0.8
        8      : 0, #0.8-0.9
        9      : 0, #0.9-1.0
        'diff' : 0  #Number of pixel with contrast (Forest-Desert, Forest-Cities, Forest-Soil) #???
    }

    shape = numpy.shape(ndviArray) #Map values from 0:255 to 0:1
    temp = ndviArray / 255.0
    noFPixel = 1.0 * shape[0] * shape[1] #Calculate the number of pixels

    for idx, el in enumerate(numpy.histogram(temp[1:-1, 1:-1], bins = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])[0]):
        NDVIGrad[idx] = el / noFPixel

    diff10 = numpy.where((temp - numpy.roll(temp, shift = 1, axis = 0)) > diffThreshold, 1, 0)
    diff11 = numpy.where((temp - numpy.roll(temp, shift = 1, axis = 1)) > diffThreshold, 1, 0)
    diffArrayThr = numpy.where(temp > pixelThreshold, diff10 + diff11, 0)

    NDVIGrad['diff'] = (diffArrayThr[1:-1,1:-1].sum()) / noFPixel

    return NDVIGrad