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
        yield {'x' : x, 'y' : y, 'z' : z}
        
        
        
magnetometer_values = get_compass_raw_Dummy()
while True:
    try:
	mag = next(magnetometer_values) 
	data_logger.info('%f , %f , %f', round(mag['x'], 4), round(mag['y'], 4), round(mag['z'], 4))
    except StopIteration:
	break
