def getIssPos():
       file=open("./data02.csv","r")
       issPositions=[]
       i=0

       for riga in file:
              piecesR=riga[0:-1].split(",")
              if (i%2==0) and not (piecesR[2]==-1):

                     issPositions.append((float(piecesR[15]),float(piecesR[16])))
              i+=1
       return issPositions