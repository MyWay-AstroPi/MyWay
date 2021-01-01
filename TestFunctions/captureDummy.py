import cv2 as cv
import numpy as np

'''
def BGR_to_GRAY(bgr_list):
    bgr_list = np.uint8([[[bgr_list[0],bgr_list[1],bgr_list[2]]]])

    gray_avg_Values = cv.cvtColor(bgr_list,cv.COLOR_BGR2GRAY)
    gray_avg_Values = np.squeeze(gray_avg_Values)

    return gray_avg_Values


def avg_center(img, percentage = 10 ,threshold_list = [0,0,0]):

    bgr_list = []
    height,width,_ = img.shape

    centerX = (width // 2 ) 
    centerY = (height // 2)                                                                  

    #RightBorder 
    XRB = centerX + ((width * percentage)//200)                    
    #LeftBorder
    XLB = centerX - ((width * percentage)//200)
    #TopBorder
    YTB = centerY + ((height * percentage)//200)
    #BottomBorder
    YBB = centerY - ((height * percentage)//200)
    
    for x in range(XLB,XRB):
        for y in range(YBB,YTB):
            bgr_list.append(img[y,x])

    numpy_bgr_array = np.array(bgr_list)
    average_value = np.average(numpy_bgr_array,axis=0)

    average_value = average_value.astype(int)

    return average_value,True 
'''

def captureDummy():
    path = ""

    for i in range(1,10):
        path = "C:\\Users\\andre\\Desktop\\astropi\\picamera\\foto\\" + str(i) + ".jpg"
        img = cv.imread(path)
        #average_value,_ = avg_center(img)
        #average_value = BGR_to_GRAY(average_value)  
        yield img #average_value

capture = captureDummy()
while True:
	try:
		print(next(capture))
	except StopIteration:
		break
