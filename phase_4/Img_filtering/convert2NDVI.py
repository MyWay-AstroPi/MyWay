import numpy as np
import cv2 as cv
import os

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


def main():
    try:
        os.mkdir('ndvi_photos')
        os.mkdir('filter_photos')
    except Exception:
        pass
    for i in range(1,35):
        img = cv.imread(f'./Experiment_Data/img_{str(i).zfill(3)}.jpg')
        img = calculateNDVI(img)
        cv.imwrite(f"./ndvi_photos/img_{str(i).zfill(3)}.jpg", img)
        img = cv.applyColorMap(img, cv.COLORMAP_RAINBOW)
        cv.imwrite(f"./filter_photos/img_{str(i).zfill(3)}.jpg", img)
        print(f'converted images: {i}')

    for i in range(1,63):
        img = cv.imread(f'./Experiment_Data/secondary_img_{str(i).zfill(3)}.jpg')
        img = calculateNDVI(img)
        cv.imwrite(f"./ndvi_photos/secondary_img_{str(i).zfill(3)}.jpg", img)
        img = cv.applyColorMap(img, cv.COLORMAP_RAINBOW)
        cv.imwrite(f"./filter_photos/secondary_img_{str(i).zfill(3)}.jpg", img)
        print(f'converted secondary images: {i}')

if __name__ == "__main__":
    main()