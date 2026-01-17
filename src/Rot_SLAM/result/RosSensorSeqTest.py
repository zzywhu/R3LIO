#-*-coding:utf-8-*-
from matplotlib import pyplot as plt
from os import listdir
from operator import itemgetter
from math import sqrt,cos,sin,tan,atan
import numpy as np
import csv
import os

imuPath = "D:\\wcy\\data\\0726\\室外\\2023-07-26-14-15-14\\imuSeq"
motorPath = "D:\\wcy\\data\\0726\\室外\\2023-07-26-14-15-14\\motorSeq"
lidPath = "D:\\wcy\\data\\0726\\室外\\2023-07-26-14-15-14\\lidSeq"

imufile = open(imuPath)
motorfile = open(motorPath)
lidfile = open(lidPath)

imuSeqList=[]
motorSeqList=[]
lidSeqList=[]
imuTimeList=[]
motorTimeList=[]
lidTimeList=[]

line = imufile.readline()
while line:
    lineStrList = line[:-1].split(" ")
    imuSeqList.append(float(lineStrList[0]))
    imuTimeList.append(float(lineStrList[2]))

    line = imufile.readline()

line = motorfile.readline()
while line:
    lineStrList = line[:-1].split(" ")
    motorSeqList.append(float(lineStrList[0]))
    motorTimeList.append(float(lineStrList[2]))

    line = motorfile.readline()

line = lidfile.readline()
while line:
    lineStrList = line[:-1].split(" ")
    lidSeqList.append(float(lineStrList[0]))
    lidTimeList.append(float(lineStrList[2]))

    line = lidfile.readline()

plt.xlabel('Sequence')
plt.ylabel('Sensor Timestamp')

plt.plot(imuSeqList, imuTimeList, color='red', linewidth=1, label="imu time")
plt.plot(motorSeqList, motorTimeList, color='green', linewidth=1, label="motor time")
plt.plot(lidSeqList, lidTimeList, color='blue', linewidth=1, label="lid time")
plt.legend()
plt.show()
