#-*-coding:utf-8-*-
from matplotlib import pyplot as plt
from os import listdir
from operator import itemgetter
from math import sqrt,cos,sin,tan,atan
import numpy as np
import csv
import os

timePath = "/home/w/code/rigelslam-release/src/result/raw/timeCost.txt"

timefile = open(timePath)

time1List=[]
time2List=[]
time3List=[]
time4List=[]

nLine=0
line = timefile.readline()
while line:
    if nLine==0:
        nLine += 1
        continue
    lineStrList = line[:-1].split(" ")
    time1List.append(float(lineStrList[0]))
    time2List.append(float(lineStrList[1]))
    time3List.append(float(lineStrList[2]))
    time4List.append(float(lineStrList[3]))
    line = timefile.readline()
    nLine+=1

plt.xlabel('frame id')
plt.ylabel('time/ms')
plt.plot(range(0,len(time1List)), time1List, color='red', linewidth=1, label="Imu process")
plt.plot(range(0,len(time2List)), time2List, color='yellow', linewidth=1, label="Downsample cloud")
plt.plot(range(0,len(time3List)), time3List, color='blue', linewidth=1, label="Update state")
plt.plot(range(0,len(time4List)), time4List, color='green', linewidth=1, label="Update map")
plt.legend()
plt.show()




plt.show()