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
from torch.autograd import Variable
from model import *

class data_load(Dataset):

    def __init__(self, data_dir, unbalancing_rate, mode,sequence_length):
        self.data_file_list = getFileNames(data_dir,unbalancing_rate, mode)
        self.data_dir = data_dir
        self.mode = mode
        self.sequence_length = sequence_length
        self.data,self.label = readDataFile(self.data_dir,self.data_file_list,sequence_length), parseFileName(self.data_file_list)
    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])


class oversample_data_load(Dataset):
    
    def __init__(self, data_dir, unbalancing_rate, mode, sequence_length):
        self.data_file_list = getFileNames(data_dir,unbalancing_rate, mode)
        self.data_dir = data_dir
        self.mode = mode
        self.data,self.label = oversampling(readDataFile(self.data_dir,self.data_file_list,sequence_length),parseFileName(self.data_file_list),sequence_length)
    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])


class lstm_retrain_dataload(Dataset):
    
    def __init__(self, data_dir, unbalancing_rate, mode,sequence_length, net, generator, device, if_test):
        self.data_file_list = getFileNames(data_dir,unbalancing_rate, mode)
        self.data_dir = data_dir
        self.mode = mode
        self.sequence_length = sequence_length
        self.data,self.label = feature_extracted(readDataFile(self.data_dir,self.data_file_list,sequence_length), parseFileName(self.data_file_list), net=net, generator=generator, device=device, if_test=if_test)
    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])




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

        if mode == 'lstm' or mode == 'weight_balancing' or mode == 'oversampling' or mode == 'lstm_retrain':
        # unbalancing rate에 따른 dataset unbalancing 조절

            unbalc_list = [0,1,2,3,4,5]
            
            for i in unbalc_list:
                gestureId_list[i] = gestureId_list[i][::int(1/unbalancing_rate)]
            
            for i in range(len(gestureId_list)):
                for j in range(len(gestureId_list[i])):
                    csv_list.append(gestureId_list[i][j])
        
        elif mode == 'feature_gan' :

            for i in range(12):
                gestureId_list[i] = gestureId_list[i][::int(1/unbalancing_rate)]
                
                for j in range(len(gestureId_list[i])):
                    csv_list.append(gestureId_list[i][j])
        # elif mode == 'lstm_retrain':
            
        #     unbalc_list = [0,1,2,3,4,5]
            
        #     for i in unbalc_list:
        #         gestureId_list[i] = gestureId_list[i][::int(1/unbalancing_rate)]
                
        #         for j in range(len(gestureId_list[i])):
        #             csv_list.append(gestureId_list[i][j])
    
    else:
        for i in range(len(gestureId_list)):
            for j in range(len(gestureId_list[i])):
                csv_list.append(gestureId_list[i][j])
    
    return csv_list

def parseFileName(data_list):
    
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

    label = [0]*12
    for i in range(len(label_list)):
        label[label_list[i]]+=1
    
    
    print(label)
    return label_list

def readDataFile(data_dir, dataFile,sequence_length):
    
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
            
            check = []
            for j in range(60):
                sum=0
                for k in range(i,i+A):
                    sum+=data_list[k][j]
                check.append(sum/A)
                
            index.append(check)
        data_list = index
        
        all_data_list.append(data_list)
    
    
    all_data_list = np.array(all_data_list)
    print(all_data_list.shape)

    return all_data_list



def oversampling(data_list,label_list,sequence_length):
    
    data_resample = []
    label_resample = []
    # train데이터를 넣어 복제함
    
    data_list = torch.tensor(data_list)
    label_list = torch.tensor(label_list)

    for i in range(sequence_length):
        data = data_list[:,i,:]
        sm = SMOTE(sampling_strategy='auto', k_neighbors=3, kind='regular')
        data, label = sm.fit_sample(data,label_list)

        data_resample.append(data)

    data_resample = np.transpose(data_resample, axes=(1, 0, 2))
    data_resample = torch.tensor(data_resample)


    return data_resample, torch.tensor(label)


def feature_extracted(data_list,label_list, net, generator, if_test, device):
    
    data_list = torch.tensor(data_list).to(device=device,dtype=torch.float32)
    
    if if_test == False:
        
        feature = net.feature_extraction(data_list)
        feature = feature.tolist()

        label = [0]*12
        
        unbal_class = [0,1,2,3,4,5]
        
        for i in range(len(label_list)):
            label[label_list[i]]+=1
       
        for i in range(2*(label[11]-label[0])):
            for j in unbal_class:
                z = Variable(torch.randn(1, 10).to(device))
                generated_feature = generator(z,torch.tensor(j).to(device))
                generated_feature = generated_feature.reshape(-1)
                generated_feature = generated_feature.tolist()
                
                feature.append(generated_feature)
                label_list.append(j)
        
        
                
        
        feature = np.array(feature)
       
        label = [0]*12
        for i in range(len(label_list)):
            label[label_list[i]]+=1

        print(label)
        
    else:
        feature = net.feature_extraction(data_list)
        feature = feature.tolist()
        
    return feature, label_list

    
    
    
    
    
    
    