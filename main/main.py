

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
import math
import reverse_geocoder as rg
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
RADIUS_CAMERA_VISION = 417.777/2
EARTH_RADIUS = 6371
SLEEP_TIME = 5

# Latest TLE data for ISS location
name = 'ISS (ZARYA)'
l1   = '1 25544U 98067A   21036.54866870  .00000473  00000-0  16759-4 0  9992'
l2   = '2 25544  51.6462 273.8205 0002445 338.1220  96.7186 15.48939397268198'
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
def isInCamera(issPos, cityPos):
        lat_alfa = math.pi * issPos[0] / 180;
        lat_beta = math.pi * cityPos[0] / 180;
        lon_alfa = math.pi * issPos[1] / 180;
        lon_beta = math.pi * cityPos[1] / 180;
        fi = abs(lon_alfa - lon_beta)
        p = math.acos(math.sin(lat_beta) * math.sin(lat_alfa) + 
               math.cos(lat_beta) * math.cos(lat_alfa) * math.cos(fi))
        distance = p * EARTH_RADIUS
        return distance <= RADIUS_CAMERA_VISION

def get_latlon():
    iss.compute()
    return (iss.sublat/ephem.degree, iss.sublong/ephem.degree)

def getMagnetometer():
    magnetometer_values = sh.get_compass_raw()
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
    secondary_photo_counter = 1

    info_logger.info('Starting the experiment')

    

    # This will loop for 178 minutes
    while (now_time < start_time + datetime.timedelta(minutes=178)):
        
            
        pos = get_latlon()
        location = rg.search(pos)

        try:
            # Take a pic
            cam.capture(rawCapture, format='bgr')

            # Save the pic in array like format in order to check if it is day
            img = rawCapture.array
        except Exception as e:
            info_logger.info(e)
        
        accellerometer = sh.get_accelerometer()
            
        mag = getMagnetometer()

        # Check if it's day
        citiesInPicture = []
        inCam = False
        for city in location:
            if isInCamera(pos,(float(city['lat']),float(city['lon']))):
                inCam = True
                citiesInPicture.append(city['name'])
        take_pic = is_day(img, size_percentage=SIZE_PERCENTAGE, min_threshold=MIN_GREY_COLOR_VALUE)
        
        if take_pic:
            ndvi = calculateNDVI(img)
            ndvi_stats = calculate_statistics(ndvi)
                
            if inCam:
                file_name = dir_path + "/img_" + str(photo_counter).zfill(3) + ".jpg"
                data_logger.info('%s, %s, %f, %f, %s, %s, %s, %s, %s, %s', photo_counter, ','.join([str(round(v, 4)) for v in ndvi_stats.values()]), pos[0],pos[1], mag['x'], mag['y'], mag['z'],accellerometer['roll'],accellerometer['pitch'],accellerometer['yaw'])
                info_logger.info('Saved ndvi photos: %s', photo_counter)
                photo_counter += 1
            else:
                file_name = dir_path + "/secondary_img_" + str(photo_counter).zfill(3) + ".jpg"
                data_logger.info('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s',secondary_photo_counter, ','.join([str(round(v, 4)) for v in ndvi_stats.values()]),pos[0], pos[1], mag['x'], mag['y'], mag['z'],accellerometer['roll'],accellerometer['pitch'],accellerometer['yaw'])
                info_logger.info('Saved secondary photos: %s', secondary_photo_counter)
                secondary_photo_counter += 1
            cv.imwrite(file_name,img)                               
        else:
            data_logger.info('-1, -, -, -, -, -, -, -, -, -, %f, %f, %s, %s, %s, %s, %s, %s', pos[0],pos[1], mag['x'], mag['y'], mag['z'],accellerometer['roll'],accellerometer['pitch'],accellerometer['yaw'])
               
            
        for city in location:
            if city['name'] in citiesInPicture:
                data_logger.info("%s, %s, %s, %s", photo_counter, city['name'],city['lat'],city['lon'])
            else:
                data_logger.info("%s, %s, %s, %s", secondary_photo_counter, city['name'],city['lat'],city['lon'])
        # It is necessary to take the next pic
        rawCapture.truncate(0)
            
        sleep(SLEEP_TIME)

        try:
            # Update the current time
            now_time = datetime.datetime.now()
        except Exception as e:
            info_logger.info(e)

    info_logger.info('End of the experiment')

if __name__ == "__main__":
    run()
