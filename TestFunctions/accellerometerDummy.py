def get_accellerometerDummy():
    data = open("data.csv", "r").readlines()

    #Accelerometer x y z raw data in Gs

    for line in data:
        yield {"x": line[0], "y": line[1], "z": line[2]}
        
acc = get_accellerometerDummy()
while True:
  try:
    result = next(acc)
  except StopIteration:
    break
