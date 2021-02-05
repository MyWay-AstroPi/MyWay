'''
Woods Orbit Observer and Deforestation Sensor Project
by WOOD Mission Team
Github: https://github.com/FrancescoGiraud0/WOODProject

Python 3 algorithm that run for about 3 hours with the porpuse to
take near infrared (NIR) pictures of the Earth, record some data like
latitude, longitude and magnetometer values and try to get some information
about vegetation, in particular we would try to measure the anthropogenic
impact by using a ML algorithm.
We use an unsupervisioned algorithm on board because we didn't have any pictures
of Earth taked by NoIR Camera + blue filter that permitted us to train a supervisioned
algorithm on it.
This algorithm have the function calculateNDVI() that computes the NDVI (Normalized
Difference Vegetation Index) of every pixels of each photo taken.
The NDVI permits to assessing whether or not the target being observed contains
live green vegetation.
The function calculate_statistics() generates a dictionary by counting the
number (in percentage) of pixels for every NDVI value mapped from 0.0 to 1.0 and 
a 'diff' value that is a counter (in percentage) of low vegetation pixels near
every pixels with high vegetation (NDVI value >= 0.7).
When the algorithm saved at least 50 photos, the dictionaries will be analized by
KMeans clustering algorithm.
'''

# MODULES
# -------------------------------
import logging
import logzero
from sense_hat import SenseHat
import ephem
from picamera import PiCamera
from picamera.array import PiRGBArray
import datetime
from time import sleep
import os
import cv2 as cv
import numpy as np
from math import degrees
# -------------------------------

# SETUP
# -------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__)) # Path of this python file

CAM_RESOLUTION = (2592,1944)    # Camera resoultion
CAM_FRAMERATE  = 15             # Camera Framerate
DIFF_THRESHOLD = 0.3            # Minimun threshold of contrast (Forest-Desert, Forest-Cities, Forest-Soil)
PIXEL_THRESHOLD = 0.6           # Divide by 10, is the minimun threshold for the contrast 
SIZE_PERCENTAGE = 30            # Percentage area of the picture starting from the center to apply is_day function
MIN_GREY_COLOR_VALUE = 70       # Minimun color value to save the photo
ML_MIN_N_OF_SAMPLES = 50        # Minimun pictures number to start the machine learning algorithm
CYCLE_TIME = 7                  # Cycle time in seconds

# Latest TLE data for ISS location
name = 'ISS (ZARYA)'
l1   = '1 25544U 98067A   20016.35580316  .00000752  00000-0  21465-4 0  9996'
l2   = '2 25544  51.6452  24.6741 0004961 136.6310 355.9024 15.49566400208322'
iss  = ephem.readtle(name, l1, l2)

# Connect to the Sense Hat
sh = SenseHat()

# Set a custom formatter for information log
info_formatter = logging.Formatter('%(name)s - %(asctime)-15s - %(levelname)s: %(message)s')
# Set a custom formatter for data log
data_formatter = logging.Formatter('%(name)s , %(asctime)-15s , %(message)s')
# Logger objects creation
info_logger    = logzero.setup_logger(name='info_logger', logfile=dir_path+'/data01.csv', formatter=info_formatter)
data_logger    = logzero.setup_logger(name='data_logger', logfile=dir_path+'/data02.csv', formatter=data_formatter)

# Set up the camera
cam = PiCamera()
# Set the resolution
cam.resolution = CAM_RESOLUTION
# Set the framerate
cam.framerate = CAM_FRAMERATE

# Set rawCapture
rawCapture = PiRGBArray(cam, size = CAM_RESOLUTION)
#--------------------------------

# FUNCTIONS
#--------------------------------
def get_latlon():
    iss.compute()
    return (iss.sublat/ephem.degree, iss.sublong/ephem.degree)

def getMagnetometer():
    magnetometer_values = sensor.get_compass_raw()
    mag_x, mag_y, mag_z = magnetometer_values['x'], magnetometer_values['y'], magnetometer_values['z']
    return {'x' : mag_x, 'y' : mag_y, 'z' : mag_z}


def is_day(img, size_percentage=30, min_threshold=80):
    '''
    Function that return true if in the center size percentage of the photo
    (converted to gray color scale) the average color value is more bright 
    than min_threshold (so, more simply, if it's day).
    '''

    # Get image size
    height, width, _ = img.shape

    # Calculate center coordinate
    centerX = (width // 2 )
    centerY = (height // 2)

    # Calculate RightBorder 
    XRB = centerX + ((width * size_percentage) // 200)                    
    # Calculate LeftBorder
    XLB = centerX - ((width * size_percentage) // 200)
    # Calculate TopBorder
    YTB = centerY + ((height * size_percentage) // 200)
    # Calculate BottomBorder
    YBB = centerY - ((height * size_percentage) // 200)

    bgr_list = []

    # Creation of a list of BGR values for every pixel
    for x in range(XLB,XRB):
        for y in range(YBB,YTB):
            bgr_list.append(img[y,x]) # Append the BGR value to the list

    # Convert bgr_list in a numpy array
    numpy_bgr_array = np.array(bgr_list)
    # Calculate the average value of blue, green and red
    average_value = np.average(numpy_bgr_array,axis=0)

    # Convert the type of datas
    average_value = average_value.astype(int)

    # Map values in uint8 format type
    average_value = np.uint8([[[average_value[0],average_value[1],average_value[2]]]]) 

    # Convert the color from BGR to Grayscale
    gray_avg_value = cv.cvtColor(average_value, cv.COLOR_BGR2GRAY)
    #remove single-dimensional entries from the shape of the numpy array
    gray_avg_value = np.squeeze(gray_avg_value)

    # Return True if the gray_avg value
    return gray_avg_value >= min_threshold

def contrast_stretch(im):
    '''
    Performs a simple contrast stretch of the given image, from 5-100%.
    '''
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 100)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out
 
def calculateNDVI(image):
    '''
    This function calculates the NDVI (Normalized Difference
    Vegetation Index) for each pixel of the photo and collect
    these values in "ndvi" numpy array.
    '''
    # Extract bgr values
    b, _, r = cv.split(image)
    bottom = (r.astype(float) + b.astype(float))
    # Change zeros of bottom array  
    # (to make sure to not divide by zero)
    bottom[bottom == 0] = 0.0000000000001

    # Calculate NDVI value of each pixel
    ndvi = (r.astype(float) - b) / bottom

    ndvi = contrast_stretch(ndvi)
    ndvi = ndvi.astype(np.uint8)
    
    return ndvi

def calculate_statistics(ndvi_array, pixel_threshold=PIXEL_THRESHOLD, diff_threshold=DIFF_THRESHOLD):
    '''
    This function generate a dictionary counting the percentage of pixels for every
    NDVI graduations (keys of the dictionary) of a numpy array made of NDVI values
    (a value for every pixel).
    This function also computes the 'diff' value, it is the percentage of pixels with
    a low NDVI (less vegetation) near every high NDVI pixels (every pixel with more
    than pixel_threshold NDVI value).
    '''
    NDVIGraduation = {
        0 : 0, # <0.1
        1 : 0, # 0.1-0.2
        2 : 0, # 0.2-0.3
        3 : 0, # 0.3-0.4
        4 : 0, # 0.4-0.5
        5 : 0, # 0.5-0.6
        6 : 0, # 0.6-0.7
        7 : 0, # 0.7-0.8
        8 : 0, # 0.8-0.9
        9 : 0, # 0.9-1.0
        'diff' : 0 # Number of pixel with contrast (Forest-Desert, Forest-Cities, Forest-Soil)
    }

    shape = np.shape(ndvi_array)
    # Map values from 0-255 to 0-1
    temp = ndvi_array / 255.0
    # Calculate the number of pixels
    nofpixel = 1.0 * shape[0] * shape[1]
    
    for i, val in enumerate(np.histogram(temp[1:-1,1:-1], bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])[0]):
        NDVIGraduation[i] = val/nofpixel
     
    diff_10 = np.where((temp - np.roll(temp,shift=1,axis=0)) > diff_threshold,1,0)
    diff_11 = np.where((temp - np.roll(temp,shift=1,axis=1)) > diff_threshold,1,0)
    diff_array_thr = np.where( temp>pixel_threshold, diff_10+diff_11, 0)
    
    NDVIGraduation['diff'] = (diff_array_thr[1:-1,1:-1].sum())/nofpixel

    return NDVIGraduation
#--------------------------------

def run():

    # Creation of a datetime variable to store the start time
    start_time = datetime.datetime.now()
    # Creation of a datetime variable to store the current time
    # (these will be almost the same at the start)
    now_time = datetime.datetime.now()
    
    # Counter to store the number of saved photos
    photo_counter = 1

    info_logger.info('Starting the experiment')

    # Create an numpy array with 11 columns (11 is the number of the keys of the dictionary)
    X_data = np.empty((0,11), float)

    try:
        os.mkdir("photo_ndvi")
    except Exception:
        pass

    # This will loop for 178 minutes
    while (now_time < start_time + datetime.timedelta(minutes=178)):
        try:
            
            pos = get_latlon()
			location = rg.search(pos)

            # Take a pic
            cam.capture(rawCapture, format='bgr')

            # Save the pic in array like format in order to check if it is day
            img = rawCapture.array

            magnetometer_values = getMagnetometer()

            # Check if it's day
            take_pic = is_day(image, size_percentage=SIZE_PERCENTAGE, min_threshold=MIN_GREY_COLOR_VALUE)
			if takepic:
				for city in location:
					takepic = takepic or isInCamera(pos,(float(city['lat']),float(city['lon'])))

            if takepic:
				ndvi = calculateNDVI(img)
                ndvi_stats = calculate_statistics(ndvi)
                cv.imwrite(file_name,ndvi)
				file_name = dir_path + "/foto_ndvi/img_" + str(photo_counter).zfill(3) + ".jpg"
				info_logger.info('Saved photos: %s', photo_counter)
				data_logger.info('%s, %s, %f, %f, %s, %s, %s, %s, %s, %s', photo_counter, ','.join([str(round(v, 4)) for v in ndvi_stats.values()]), pos[0],pos[1], mag['x'], mag['y'], mag['z'][:-1],acc['x'],acc['y'],acc['z'])
				photo_counter += 1
            else:
				data_logger.info('-1, -,-,-,-,-,-,-,-,-,-, %f, %f, %s, %s, %s, %s, %s, %s',pos[0], pos[1] mag['x'], mag['y'], mag['z'][:-1],acc['x'],acc['y'],acc['z'])
			
            # It is necessary to take the next pic
            rawCapture.truncate(0)

            # Update the current time
            now_time = datetime.datetime.now()

        except Exception as e:
            info_logger.error('An error occurred: %s', str(e))

    info_logger.info('End of the experiment')

if __name__ == "__main__":
    run()
