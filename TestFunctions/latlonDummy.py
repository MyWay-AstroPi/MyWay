'''
this function reads cordinates from a csv file
'''

def get_latlon_Dummy():
    readDate = open("dataCord.txt", "r").readlines()
    for line in readData:
        values = line.split(,)
        
        sublat = values[0]
        sublong = values[1]
        
        lat_value = [float(i) for i in str(sublat).split(':')]
        long_value = [float(i) for i in str(sublong).split(':')]
        yield  lat_value, long_value
        
        
cords = get_latlon_Dummy()
while True:
	lat, lon = next(cords)
	data_logger.info('%f %f %f %f %f %f',lat[0],lat[1],lat[2],lon[0],lon[1],lon[2])
