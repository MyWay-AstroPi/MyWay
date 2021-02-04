import logging
import logzero
import os
import datetime
import cv2 as cv
import numpy as np
import ephem 
import reverse_geocoder as rg
import math

DIFF_THRESHOLD = 0.3
PIXEL_THRESHOLD = 0.6
RADIUS_CAMERA_VISION = 417.777/2
EARTH_RADIUS = 6371

dir_path = os.path.dirname(os.path.realpath(__file__)) # Path of this python file

# Set a custom formatter for information log
info_formatter = logging.Formatter('%(name)s - %(asctime)-15s - %(levelname)s: %(message)s')
# Set a custom formatter for data log
data_formatter = logging.Formatter('%(name)s , %(asctime)-15s , %(message)s')
# Logger objects creation
info_logger = logzero.setup_logger(name='info_logger', logfile=dir_path+'/data01.csv', formatter=info_formatter)
data_logger = logzero.setup_logger(name='data_logger', logfile=dir_path+'/data02.csv', formatter=data_formatter)



# Latest TLE data for ISS location
name = 'ISS (ZARYA)'
l1   = '1 25544U 98067A   21032.50223113  .00000681  00000-0  20555-4 0  9991'
l2   = '2 25544  51.6455 293.8368 0002186 326.8516 209.4821 15.48930319267560'
iss  = ephem.readtle(name, l1, l2)



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


def get_latlon_Dummy():
	readData = open("dataCord.csv", "r").readlines()
	for line in readData:
		values = line.split(',')
		
		sublat = values[0]
		sublong = values[1]
		
		lat_value = [float(i) for i in str(sublat).split(':')]
		long_value = [float(i) for i in str(sublong).split(':')]
		yield  lat_value, long_value
		
		


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
		path = dir_path + "/foto/" + next(file_names)[:-1]
		img = cv.imread(path)        
		yield img



def get_compass_raw_Dummy():
	ReadData = open("dataMagnet.csv", "r").readlines()
	for elem in ReadData:
		values = elem.split(',')
		x = values[0]
		y = values[1]
		z = values[2]
		yield {'x' : x, 'y' : y, 'z' : z}
		
		  
def get_accellerometerDummy():
	data = open("dataAccellerometer.csv", "r").readlines()
	#Accelerometer x y z raw data in Gs
	for line in data:
		line = line.split(',')
		yield {"x": line[0], "y": line[1], "z": line[2]}


def main():
	start_time = datetime.datetime.now()
	now_time = datetime.datetime.now()
	
	photo_counter = 1
	
	info_logger.info('Starting the expertiment')
	data_logger.info('photo_counter, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, diff, lat[0], lat[1], lat[2], lon[0], lon[1], lon[2], magX, magY, magZ, accX, accY, accZ')
	
	cords = get_latlon_Dummy()
	capture = captureDummy()
	magnetometer_values = get_compass_raw_Dummy()
	accellerometer_values = get_accellerometerDummy()
	os.mkdir("foto_ndvi")
	
	while now_time < start_time + datetime.timedelta(minutes = 178):
		try:
			iss.compute()
			pos = (iss.sublat / ephem.degree, iss.sublong / ephem.degree)
			location = rg.search(pos)
			print(location)
			
			#get latitude, longitude
			lat,lon = next(cords)
			
			
			#get magnetometer values
			mag = next(magnetometer_values) 
						
			#get accellerometer values
			acc = next(accellerometer_values)
			#take pictures and apply the ndvi filter
			img = next(capture)
			
			
			takepic = is_day(img)
			if takepic:
				for city in location:
					takepic = takepic or isInCamera(pos,(float(city['lat']),float(city['lon'])))

			if takepic:
				ndvi = calculateNDVI(img)
				file_name = dir_path + "/foto_ndvi/img_" + str(photo_counter).zfill(3) + ".jpg"
				cv.imwrite(file_name,ndvi)
				ndvi_stats = calculate_statistics(ndvi)
				info_logger.info('Saved photos: %s', photo_counter)
				data_logger.info('%s, %s, %f, %f, %f, %f, %f, %f, %s , %s , %s, %s, %s, %s', photo_counter, ','.join([str(round(v, 4)) for v in ndvi_stats.values()]), lat[0], lat[1], lat[2], lon[0], lon[1], lon[2], mag['x'], mag['y'], mag['z'][:-1],acc['x'],acc['y'],acc['z'])
				photo_counter += 1
			
			else:
				data_logger.info('%s, -,-,-,-,-,-,-,-,-,-, %f, %f, %f, %f, %f, %f, %s , %s , %s, %s, %s, %s', '-1', lat[0], lat[1], lat[2], lon[0], lon[1], lon[2], mag['x'], mag['y'], mag['z'][:-1],acc['x'],acc['y'],acc['z'])
			
			for city in location:
				data_logger.info("citta: %s, lat: %s, lon: %s",city['name'],city['lat'],city['lon'])
			now_time = datetime.datetime.now()
			
		except StopIteration:
			info_logger.info("error")
			break
	info_logger.info("end of the experiment")

if __name__ == "__main__":
	main()
