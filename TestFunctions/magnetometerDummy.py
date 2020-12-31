'''
this function reads from a csv file the magnetometer values
'''


def get_compass_raw_Dummy():
    ReadData = open("dataMagnet.txt", "r").readlines()
    for elem in ReadData:
        values = elem.split(',')
        x = values[0]
        y = values[1]
        x = values[2]
        yield x,y,z
        
        
        
magnetometer_values = get_compass_raw_Dummy()
while True:
	mag_x, mag_y, mag_z = next(magnetometer_values) 
	data_logger.info('%s , %s , %s', round(mag_x, 4), round(mag_y, 4), round(mag_z, 4))
