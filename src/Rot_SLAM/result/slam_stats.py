#-*-coding:utf-8-*-
from matplotlib import pyplot as plt
from os import listdir
from operator import itemgetter
from math import sqrt,cos,sin,tan,atan
import numpy as np
import csv
import os
import sys

statPath = "/home/w/code/rigelslam-release/src/result/raw/stats.txt"

statfile = open(sys.argv[1])

timeList=[]
memList=[]

line = statfile.readline()
while line:
    lineStrList = line[:-1].split(",")
    timeList.append(float(lineStrList[0]))
    memList.append(float(lineStrList[1]))

    line = statfile.readline()


plt.xlabel('frame id')
plt.ylabel('process time/ms')
plt.plot(range(0,len(timeList)), timeList, color='blue', linewidth=1, label="Frame process time")
plt.legend()
plt.show()

plt.xlabel('frame id')
plt.ylabel('Total memory usage/ms')
plt.plot(range(0,3000), memList[0:3000], color='blue', linewidth=1, label="Total memory usage")
plt.legend()
plt.show()
