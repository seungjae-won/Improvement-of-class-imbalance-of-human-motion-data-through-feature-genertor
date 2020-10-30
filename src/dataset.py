from os.path import splitext
from os import listdir
import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import sys
from random import sample
import math
import re
import random
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

class data_load(Dataset):

    def __init__(self, data_dir, unbalancing_rate, mode,sequence_length):
        self.data_file_list = getFileNames(data_dir,unbalancing_rate, mode)
        self.data_dir = data_dir
        self.mode = mode
        self.sequence_length = sequence_length
    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        self.data = readDataFile(os.path.join(self.data_dir,self.data_file_list[index]),self.sequence_length)
        self.label = parseFileName(os.path.join(self.data_dir,self.data_file_list[index]), self.mode)
        return self.data, self.label

def getFileNames(input_path, unbalancing_rate, mode):
    
    input_path = glob.glob('%s/*.csv' % input_path)
    gestureId_list = [[] for _ in range(12)]
    
    for i in range(len(input_path)):
        input_path[i] = input_path[i].split("/")[-1]
        noSuffix = input_path[i].split(".")[0]
        fields = noSuffix.split("_")	
        gestureId = fields[2]				
        if (gestureId[-1] == 'A'):
            twoModalities = True
            gestureId = gestureId[:-1]
        gestureId = int(gestureId)
        gestureId_list[gestureId-1].append(input_path[i])
    
    csv_list = []
    if unbalancing_rate != 1:
        if mode == 'lstm':
        # unbalancing rate에 따른 dataset unbalancing 조절

            unbalc_list = [0,2,4,6,8,10]
            
            for i in unbalc_list:
                gestureId_list[i] = gestureId_list[i][::int(1/unbalancing_rate)]
            
            for i in range(len(gestureId_list)):
                for j in range(len(gestureId_list[i])):
                    csv_list.append(gestureId_list[i][j])
        
        else:
            unbalc_list = [0,2,4,6,8,10]
        
            for i in unbalc_list:
                gestureId_list[i] = gestureId_list[i][::int(1/unbalancing_rate)]
                
                for j in range(len(gestureId_list[i])):
                    csv_list.append(gestureId_list[i][j])
    
    else:
        for i in range(len(gestureId_list)):
            for j in range(len(gestureId_list[i])):
                csv_list.append(gestureId_list[i][j])
    
        
    return csv_list

def parseFileName(fName, mode):

    fName = fName.split("/")[-1]
    noSuffix = fName.split(".")[0]
    fields = noSuffix.split("_")	
    gestureId = fields[2]				
    if (gestureId[-1] == 'A'):
        twoModalities = True
        gestureId = gestureId[:-1]

    # gestureId_list.append(int(gestureId)-1)
    # gestureId_list = np.array(gestureId_list)
    
    if mode == 'lstm':
        label = torch.tensor(int(gestureId)-1)
    else:
        label = torch.tensor((int(gestureId)-1)//2)
    return label

def readDataFile(dataFile,sequence_length):
    contents = np.genfromtxt(dataFile, delimiter=' ')
    data = list(contents[:,1:])

    data_list = []

    for i in range(len(data)):
        data_line = []
        
        if data[i][0] == 0 and data[i][1] == 0 and data[i][2] == 0:
            continue 
        while len(data[i]) != 0:
            for _ in range(3):
                data_line.append(data[i][0])
                data[i] = np.delete(data[i],0)
            data[i] = np.delete(data[i],0)
        data_list.append(data_line)

    index = []
    A = len(data_list) // sequence_length
    for i in range(0,len(data_list),A):
        if len(index) == sequence_length:
            break
        index.append(data_list[i])
    data_list = index
    return torch.tensor(data_list)

class oversample_data_load(Dataset):
    
    def __init__(self, data_dir, unbalancing_rate, mode, sequence_length):
        self.data_file_list = getFileNames_2(data_dir,unbalancing_rate, mode)
        self.data_dir = data_dir
        self.mode = mode
        self.data,self.label = oversampling(readDataFile_2(self.data_dir,self.data_file_list,sequence_length),parseFileName_2(self.data_file_list),sequence_length)
    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])

def getFileNames_2(input_path, unbalancing_rate, mode):
    
    input_path = glob.glob('%s/*.csv' % input_path)
    gestureId_list = [[] for _ in range(12)]
    
    for i in range(len(input_path)):
        input_path[i] = input_path[i].split("/")[-1]
        noSuffix = input_path[i].split(".")[0]
        fields = noSuffix.split("_")	
        gestureId = fields[2]				
        if (gestureId[-1] == 'A'):
            twoModalities = True
            gestureId = gestureId[:-1]
        gestureId = int(gestureId)
        gestureId_list[gestureId-1].append(input_path[i])
    
    csv_list = []
    if unbalancing_rate != 1:
        if mode == 'lstm':
            unbalc_list = [0,2,4,6,8,10]
            
            for i in unbalc_list:
                gestureId_list[i] = gestureId_list[i][::int(1/unbalancing_rate)]
            
            for i in range(len(gestureId_list)):
                for j in range(len(gestureId_list[i])):
                    csv_list.append(gestureId_list[i][j])
        
        else:
            unbalc_list = [0,2,4,6,8,10]
        
            for i in unbalc_list:
                gestureId_list[i] = gestureId_list[i][::int(1/unbalancing_rate)]
                
                for j in range(len(gestureId_list[i])):
                    csv_list.append(gestureId_list[i][j])
    
    else:
        for i in range(len(gestureId_list)):
            for j in range(len(gestureId_list[i])):
                csv_list.append(gestureId_list[i][j])
    
        
    return csv_list

def parseFileName_2(data_list):

    label_list = []
    
    for fName in data_list:
        fName = fName.split("/")[-1]
        noSuffix = fName.split(".")[0]
        fields = noSuffix.split("_")	
        gestureId = fields[2]				
        if (gestureId[-1] == 'A'):
            twoModalities = True
            gestureId = gestureId[:-1]

        label = (int(gestureId)-1)
        
        label_list.append(label)

    return label_list

def readDataFile_2(data_dir, dataFile,sequence_length):
    
    all_data_list = []
    for fName in dataFile:
        
        data_path = os.path.join(data_dir,fName)
        contents = np.genfromtxt(data_path, delimiter=' ')
        data = list(contents[:,1:])

        data_list = []

        for i in range(len(data)):
            data_line = []
            
            if data[i][0] == 0 and data[i][1] == 0 and data[i][2] == 0:
                continue 
            while len(data[i]) != 0:
                for _ in range(3):
                    data_line.append(data[i][0])
                    data[i] = np.delete(data[i],0)
                data[i] = np.delete(data[i],0)
            data_list.append(data_line)

        index = []
        A = len(data_list) // sequence_length
        
        for i in range(0,len(data_list),A):
            if len(index) == sequence_length:
                break
            index.append(data_list[i])
        data_list = index
        
        all_data_list.append(data_list)
    return all_data_list


def oversampling(data_list,label_list,sequence_length):
    
    data_resample = []
    label_resample = []
    # train데이터를 넣어 복제함
    
    data_list = torch.tensor(data_list)
    label_list = torch.tensor(label_list)
    
    for i in range(sequence_length):
        data = data_list[:,i,:]
        sm = SMOTE(ratio='auto', kind='regular')
        data, label = sm.fit_sample(data,label_list)

        data_resample.append(data)

    data_resample = np.transpose(data_resample, axes=(1, 0, 2))
    data_resample = torch.tensor(data_resample)

    return data_resample, torch.tensor(label)

def oversampling(data_list,label_list,sequence_length):
    
    data_resample = []
    label_resample = []
    # train데이터를 넣어 복제함
    
    data_list = np.array(data_list)
    
    
    data_list = torch.tensor(data_list)
    label_list = torch.tensor(label_list)
    
    for i in range(sequence_length):
        data = data_list[:,i,:]
        sm = SMOTE(ratio='auto', kind='regular')
        data, label = sm.fit_sample(data,label_list)

        data_resample.append(data)

    data_resample = np.transpose(data_resample, axes=(1, 0, 2))
    data_resample = torch.tensor(data_resample)

    return data_resample, torch.tensor(label)