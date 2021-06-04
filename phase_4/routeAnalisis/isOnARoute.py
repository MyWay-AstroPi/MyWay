import math 
import routes


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


def isInCamera(issPos, routePos):
       lat_alfa = math.pi * issPos[0] / 180;
       lat_beta = math.pi * float(routePos[0]) / 180;
       lon_alfa = math.pi * issPos[1] / 180;
       lon_beta = math.pi * float(routePos[1]) / 180;
        
       fi = abs(lon_alfa - lon_beta)
       p = math.acos(math.sin(lat_beta) * math.sin(lat_alfa) + math.cos(lat_beta) * math.cos(lat_alfa) * math.cos(fi))
       distance = p * EARTH_RADIUS
       return distance <= RADIUS_CAMERA_VISION

def isonThisRoute(issPos, route):
       for i in route:
              if isInCamera(issPos, i):
                     return issPos 
       return False

def isOnARoute(issPos):
       goon=[]
       lstRoutes=[routes.blackBelliedPlover_19, routes.blackBelliedPlover_20, 
       routes.blackBelliedPlover_7, routes.blackBelliedPlover_8, routes.longBilledCurlew_141772, routes.longBilledCurlew_141773,
       routes.longBilledCurlew_154067, routes.longBilledCurlew_154069, routes.longBilledCurlew_154072, routes.longBilledCurlew_154074,
       routes.pacificLoon_alpine01, routes.pacificLoon_alpine03, routes.pacificLoon_YK03, routes.pacificLoon_YK08, 
       routes.swainsonHawk_SW16, routes.swainsonHawk_SW17, routes.swainsonHawk_SW18]

       lstNameRoutes=["blackBelliedPlover_19 ","blackBelliedPlover_20 ","blackBelliedPlover_7 ","blackBelliedPlover_8 ","longBilledCurlew_141772 ","longBilledCurlew_141773 ","longBilledCurlew_154067 ","longBilledCurlew_154069 ","longBilledCurlew_154072 ","longBilledCurlew_154074 ","pacificLoon_alpine01 ","pacificLoon_alpine03 ","pacificLoon_YK03 ","pacificLoon_YK08 ","swainsonHawk_SW16 ","swainsonHawk_SW17 ","swainsonHawk_SW18"]
       
       for cont, route in enumerate(lstRoutes):
              risult=isonThisRoute(issPos, route)
              if risult:
                     goon.append((lstNameRoutes[cont],risult))
              
       return goon