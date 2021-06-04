
def findCross():
    cross=[]
    for i in getIssPos():
        cross.append(isOnARoute(i)) 
    file=open('./incroci.csv', 'w')
    for i in cross:
        if str(i)=='[]':
                continue
        file.write(str(i)+"\n") 
    print(cross)