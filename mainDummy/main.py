import logging
import logzero
import os
import datetime
import cv2 as cv
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__)) # Path of this python file

# Set a custom formatter for information log
info_formatter = logging.Formatter('%(name)s - %(asctime)-15s - %(levelname)s: %(message)s')
# Set a custom formatter for data log
data_formatter = logging.Formatter('%(name)s , %(asctime)-15s , %(message)s')
# Logger objects creation
info_logger = logzero.setup_logger(name='info_logger', logfile=dir_path+'/data01.csv', formatter=info_formatter)
data_logger = logzero.setup_logger(name='data_logger', logfile=dir_path+'/data02.csv', formatter=data_formatter)




def get_latlon_Dummy():
    readData = open("dataCord.txt", "r").readlines()
    for line in readData:
        values = line.split(',')
        
        sublat = values[0]
        sublong = values[1]
        
        lat_value = [float(i) for i in str(sublat).split(':')]
        long_value = [float(i) for i in str(sublong).split(':')]
        yield  lat_value, long_value
        
        
DIFF_THRESHOLD = 0.3
PIXEL_THRESHOLD = 0.6


'''
UNKNOWN FUNCTION
'''
def contrastStretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 100)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out


def calculateNDVI(image):
    b, _, r = cv.split(image) #Extraction of bgr valuer (ignore the g value)
    bottom = r.astype('float') + b.astype('float')
    bottom[bottom == 0] = 0.0000000000001 #Change 0s in bottom array (exclude the division by 0)
    ndvi = (r.astype('float') - b) / bottom #Calculate NDVI value of each pixel

    ndvi = contrastStretch(ndvi) #???
    ndvi = ndvi.astype(np.uint8) #Conversion to numpy array

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

    shape = np.shape(ndviArray) #Map values from 0:255 to 0:1
    temp = ndviArray / 255.0
    noFPixel = 1.0 * shape[0] * shape[1] #Calculate the number of pixels

    for idx, el in enumerate(np.histogram(temp[1:-1, 1:-1], bins = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])[0]):
        NDVIGrad[idx] = el / noFPixel

    diff10 = np.where((temp - np.roll(temp, shift = 1, axis = 0)) > diffThreshold, 1, 0)
    diff11 = np.where((temp - np.roll(temp, shift = 1, axis = 1)) > diffThreshold, 1, 0)
    diffArrayThr = np.where(temp > pixelThreshold, diff10 + diff11, 0)

    NDVIGrad['diff'] = (diffArrayThr[1:-1,1:-1].sum()) / noFPixel

    return NDVIGrad


def find_file_name():
	names = open("names.txt",'r')
	for line in names:
		yield line


def captureDummy():
    path = ""
    file_names = find_file_name()
    while True:
        path = "C:\\Users\\andre\\Desktop\\astropi\\picamera\\foto\\" + next(file_names)[:-1] 
        img = cv.imread(path)        
        yield img




def run():
	start_time = datetime.datetime.now()
	now_time = datetime.datetime.now()
	
	photo_counter = 1
	
	info_logger.info('Starting the expertiment')
	
	cords = get_latlon_Dummy()
	capture = captureDummy()
	while now_time < start_time + datetime.timedelta(minutes = 178):
		try:
			
			#get latitude, longitude
			lat,lon = next(cords)
			data_logger.info('%f, %f, %f, %f, %f, %f', lat[0], lat[1], lat[2], lon[0], lon[1], lon[2])
			
			
			#take pictures and apply the ndvi filter
			photo_counter += 1
			img = next(capture)
			ndvi = calculateNDVI(img)
			file_name = dir_path + "/foto_ndvi/img_" + str(photo_counter).zfill(3) + ".jpg"
			cv.imwrite(file_name,ndvi)
			ndvi = calculate_statistics(ndvi)
			
			#take magnetometer values
			#take accellerometer values
			
			
		except StopIteration:
			break

run()
