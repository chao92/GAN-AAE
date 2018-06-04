import sys
import numpy as np
#dataset=sys.argv[1]
gan_path=sys.argv[2]
original_path=sys.argv[1]
edgeCount=0
hit=0.0
with open(original_path,"r") as fp:
    firstLine=fp.readline()
    firstLine=firstLine.strip("\n")
    firstLine=firstLine.split(" ")
    edgeCount=int(firstLine[0])
    original_labels=np.zeros((edgeCount,3))
    for line in fp:
        line=line.strip("\n")
        line=line.split(" ")
        index=int(line[0])
        for i in range(2,len(line)):
            original_labels[index][int(line[i])-1]=1

count=0
with open(gan_path,"r") as fp:
    for line in fp:
        line=line.strip("\n")
        line=int(line)
        if(original_labels[count][line]==1):
            hit+=1
        count+=1
print(hit/count)
