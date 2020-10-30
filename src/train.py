from google.colab import drive
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import glob
import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.functional as F
import torch.autograd as autograd
from model import *
from dataset import *
from util import *
import argparse
from torch.autograd import Variable
from train import *

class Train:
    
    def __init__(self, args):

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch

        self.train_data_dir = args.train_data_dir
        self.test_data_dir = args.test_data_dir
        self.lstm_ckpt_dir = args.lstm_ckpt_dir
        self.lstm_log_dir = args.lstm_log_dir
        self.oversampling_ckpt_dir = args.oversampling_ckpt_dir
        self.oversampling_log_dir = args.oversampling_log_dir
        self.weight_balancing_ckpt_dir = args.weight_balancing_ckpt_dir
        self.weight_balancing_log_dir = args.weight_balancing_log_dir
        self.feature_gan_ckpt_dir = args.feature_gan_ckpt_dir
        self.feature_gan_log_dir = args.feature_gan_log_dir
        self.lstm_retrain_ckpt_dir = args.lstm_retrain_ckpt_dir
        self.lstm_retrain_log_dir = args.lstm_retrain_log_dir

        self.sequence_length = args.sequence_length
        self.input_size = args.input_size
        self.num_lstm = args.num_lstm
        self.lstm_hidden_size = args.lstm_hidden_size
        self.fg_hidden_size = args.fg_hidden_size
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.latent_size = args.latent_size

        self.mode = args.mode
        self.unbalancing_rate = args.unbalancing_rate
        self.train_continue = args.train_continue

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def lstm(self):
        
        if self.mode == 'lstm' or self.mode == 'weight_balancing':
            train_dataset = data_load(data_dir = self.train_data_dir,
                            unbalancing_rate = self.unbalancing_rate, mode=self.mode, sequence_length=self.sequence_length)
            test_dataset = data_load(data_dir = self.test_data_dir, 
                                    unbalancing_rate = 1.0, mode='lstm', sequence_length=self.sequence_length)
            train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True)
            test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=self.batch_size, 
                                    shuffle=False)
        
        elif self.mode == 'oversampling':
            train_dataset = oversample_data_load(data_dir = self.train_data_dir,
                            unbalancing_rate = self.unbalancing_rate, mode=self.mode, sequence_length=self.sequence_length)
            test_dataset = data_load(data_dir = self.test_data_dir, 
                                    unbalancing_rate = 1.0, mode='lstm', sequence_length=self.sequence_length)
            train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True)
            test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=self.batch_size, 
                                    shuffle=False)

        fn_pred = lambda output: torch.softmax(output,dim=1)
        fn_acc = lambda pred, label:((pred.max(dim=1)[1] == label).type(torch.float)).mean()
        
        net = RNN_MODEL(input_size = self.input_size, hidden_size = self.lstm_hidden_size, batch_size = self.batch_size, 
                sequence_length = self.sequence_length, num_lstm = self.num_lstm, num_classes = self.num_classes, 
                dropout = self.dropout, device = self.device).to(self.device)
        
        if self.mode == 'lstm' or self.mode == 'oversampling':
            criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.mode =='weight_balancing':
            class_weight = [1/self.unbalancing_rate,1]*(self.num_classes//2)
            criterion = nn.CrossEntropyLoss(weight = torch.tensor(class_weight)).to(self.device)
            
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        num_train_data = len(train_loader.dataset)
        num_train_batch = np.ceil(num_train_data/self.batch_size)
        
        min_train_loss = 2
        
        if self.mode == 'lstm':
            writer = SummaryWriter(log_dir=os.path.join(self.lstm_log_dir))
        elif self.mode == 'oversampling':
            writer = SummaryWriter(log_dir=os.path.join(self.oversampling_log_dir))
        elif self.mode == 'weight_balancing':
            writer = SummaryWriter(log_dir=os.path.join(self.weight_balancing_log_dir))
            
            
        if self.train_continue == 'on':
            if self.mode == 'lstm':
                net, optim = load_model(ckpt_dir=self.lstm_ckpt_dir, net=net, optim=optimizer)
            elif self.mode == 'oversampling':
                net, optim = load_model(ckpt_dir=self.oversampling_ckpt_dir, net=net, optim=optimizer)
            elif self.mode == 'weight_balancing':
                net, optim = load_model(ckpt_dir=self.weight_balancing_ckpt_dir, net=net, optim=optimizer)
                
        for epoch in range(self.num_epoch):
            net.train()
            train_loss_arr = []
            train_acc_arr = []
            
            for i, (input_value, labels) in enumerate(train_loader):
                input_value = input_value.to(self.device,dtype=torch.float32)
                labels = labels.to(self.device)

                outputs = net(input_value)
                pred = fn_pred(outputs)
                
                optimizer.zero_grad()
                
                loss = criterion(outputs, labels)
                acc = fn_acc(pred,labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss_arr+=[loss.item()]
                train_acc_arr+=[acc.item()]
            
            with torch.no_grad():
                net.eval()
                test_loss_arr = []
                test_acc_arr = []
                
                for input_value, labels in test_loader:
                    
                    input_value = input_value.to(self.device,dtype=torch.float32)
                    labels = labels.to(self.device)
                    
                    outputs = net(input_value)
                    pred = fn_pred(outputs)
                    
                    test_loss = criterion(outputs, labels)
                    
                    test_acc = fn_acc(pred,labels)
                    
                    test_loss_arr+=[test_loss.item()]
                    test_acc_arr+=[test_acc.item()]
            
            print(self.mode, ' - Epoch [{}/{}]\ttrain_Loss: {:.4f}, train_Accuracy: {:.4f}\ttest_Loss: {:.4f}, test_Accuracy: {:.4f}' .format(epoch+1, self.num_epoch, np.mean(train_loss_arr) , np.mean(train_acc_arr), 
                                                                                                                                np.mean(test_loss_arr) , np.mean(test_acc_arr)))
            if min_train_loss > np.mean(train_loss_arr) and epoch >= 30:
                min_train_loss = np.mean(train_loss_arr)
                
                if self.mode == 'lstm':
                    save_model(ckpt_dir=self.lstm_ckpt_dir, net=net, optim=optimizer, epoch=0)
                elif self.mode == 'oversampling':
                    save_model(ckpt_dir=self.oversampling_ckpt_dir, net=net, optim=optimizer, epoch=0)
                elif self.mode == 'weight_balancing':
                    save_model(ckpt_dir=self.weight_balancing_ckpt_dir, net=net, optim=optimizer, epoch=0)
                    
            writer.add_scalar('train_loss', np.mean(train_loss_arr), epoch)
            writer.add_scalar('train_acc', np.mean(train_acc_arr), epoch)
            writer.add_scalar('test_loss', np.mean(test_loss_arr), epoch)
            writer.add_scalar('test_acc', np.mean(test_acc_arr), epoch)
            
        writer.close()
    

    def feature_gan(self):
        
        train_dataset = data_load(data_dir = self.train_data_dir,
                            unbalancing_rate = self.unbalancing_rate, mode=self.mode, sequence_length=self.sequence_length)
        test_dataset = data_load(data_dir = self.test_data_dir, 
                                unbalancing_rate = 1.0, mode='lstm', sequence_length=self.sequence_length)
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=self.batch_size, 
                                shuffle=False)
            
        gen_writer = SummaryWriter(log_dir=os.path.join(self.feature_gan_log_dir, 'generator'))
        dis_writer = SummaryWriter(log_dir=os.path.join(self.feature_gan_log_dir, 'discriminator'))
        
        
        fn_pred = lambda output: torch.softmax(output,dim=1)
        fn_acc = lambda pred, label:((pred.max(dim=1)[1] == label).type(torch.float)).mean()
        
        net_lstm = RNN_MODEL(input_size = self.input_size, hidden_size = self.lstm_hidden_size, batch_size = self.batch_size, 
                    sequence_length = self.sequence_length, num_lstm = self.num_lstm, num_classes = self.num_classes, 
                    dropout = self.dropout, device = self.device)
        
        
        _ = optim.Adam(net_lstm.parameters(), lr=self.lr)
        net_lstm, optim_= load_model(ckpt_dir=self.lstm_ckpt_dir, net=net_lstm, optim=_)
        
        net_lstm.to(self.device,dtype=torch.float32)

        G = Feautre_Generator(latent_size=self.latent_size, hidden_size=self.fg_hidden_size, output_size=self.lstm_hidden_size, num_classes=self.num_classes).to(self.device)
        D = Feature_Discriminator(feature_size=self.lstm_hidden_size,hidden_size=self.fg_hidden_size, num_classes=self.num_classes).to(self.device)
        BCE_criterion = nn.BCELoss().to(self.device)
        CE_criterion = nn.CrossEntropyLoss().to(self.device)
        
        d_optimizer = optim.Adam(D.parameters(), lr=self.lr)
        g_optimizer = optim.Adam(G.parameters(), lr=self.lr)
        
        if self.train_continue == 'on':
            G, g_optimizer = load_model(ckpt_dir=os.path.join(self.feature_gan_ckpt_dir,'generator'), net=G, optim=g_optimizer)
            D, d_optimizer = load_model(ckpt_dir=os.path.join(self.feature_gan_ckpt_dir,'discriminator'), net=D, optim=d_optimizer)
        
        G.train()
        D.train()
        
        for epoch in range(self.num_epoch):
            
            D_loss = []
            G_loss = []
            accuracy = []
            
            for i, (input_value, labels) in enumerate(train_loader):
                
                batch_size = len(labels)

                input_value = input_value.to(self.device,dtype=torch.float32)
                real_bce_labels = torch.ones(batch_size, 1).to(self.device)
                fake_bce_labels = torch.zeros(batch_size, 1).to(self.device)
                labels = labels.to(self.device)
                
                lstm_feature = net_lstm.feature_extraction(input_value)
                
                
                real_bce_outputs, real_ce_outputs = D(lstm_feature)
                
                d_real_loss = BCE_criterion(real_bce_outputs, real_bce_labels) + CE_criterion(real_ce_outputs, labels)/2
                
                z = Variable(torch.randn(batch_size, self.latent_size).to(self.device))
                gen_labels = Variable(torch.LongTensor(np.random.randint(0, self.num_classes, batch_size)).to(self.device))
                
                fake_feature = G(z, gen_labels)
            
                fake_bce_outputs, fake_ce_outputs = D(fake_feature.detach())
                
                d_fake_loss = BCE_criterion(fake_bce_outputs, fake_bce_labels) + CE_criterion(fake_ce_outputs, labels)/2
                
                d_loss = (d_real_loss + d_fake_loss)/2
                
                D_loss+=[d_loss.item()]
                
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                

                bce_outputs, ce_outputs = D(fake_feature)
                
                
                real_pred = fn_pred(real_ce_outputs)   
                real_acc = fn_acc(real_pred,labels)
                fake_pred = fn_pred(fake_ce_outputs)   
                fake_acc = fn_acc(fake_pred,gen_labels)
                
                acc = (real_acc+fake_acc)/2
                accuracy+=[acc.item()]
                
                g_loss = 0.5*(BCE_criterion(bce_outputs, real_bce_labels))+CE_criterion(ce_outputs, gen_labels)
                
                G_loss+=[g_loss.item()]
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                
            
            print('Epoch [{}/{}]\tG_loss : {:.4f}\tD_loss : {:.4f}\tAccuracy: {:.4f}' .format(epoch+1, self.num_epoch, np.mean(G_loss) , np.mean(D_loss), np.mean(accuracy)))       
                
            gen_writer.add_scalar('Generator_Loss', np.mean(G_loss), epoch)
            dis_writer.add_scalar('Discriminator_Loss', np.mean(D_loss), epoch)
            dis_writer.add_scalar('Accuracy', np.mean(accuracy), epoch)
            
            
            save_model(ckpt_dir=os.path.join(self.feature_gan_ckpt_dir, 'generator'), net=G, optim=g_optimizer, epoch=0)
            save_model(ckpt_dir=os.path.join(self.feature_gan_ckpt_dir, 'discriminator'), net=D, optim=d_optimizer, epoch=0)
            
        gen_writer.close()
        dis_writer.close()
             
    def lstm_retrain(self):
        
        train_dataset = data_load(data_dir = self.train_data_dir,
                            unbalancing_rate = self.unbalancing_rate, mode=self.mode, sequence_length=self.sequence_length)
        test_dataset = data_load(data_dir = self.test_data_dir, 
                                unbalancing_rate = 1.0, mode='lstm', sequence_length=self.sequence_length)
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=self.batch_size, 
                                shuffle=False)
            
        
        net = RNN_MODEL(input_size = self.input_size, hidden_size = self.lstm_hidden_size, batch_size = self.batch_size, 
                    sequence_length = self.sequence_length, num_lstm = self.num_lstm, num_classes = self.num_classes, 
                    dropout = self.dropout, device = self.device).to(self.device)

        criterion = nn.CrossEntropyLoss().to(self.device)

        optimizer = optim.Adam(net.parameters(), lr=self.lr)

        num_train_data = len(train_loader.dataset)
        num_train_batch = np.ceil(num_train_data/self.batch_size)
        
        min_train_loss = 2
        
        writer = SummaryWriter(log_dir=os.path.join(self.lstm_retrain_log_dir))
        
        net, optimizer = load_model(ckpt_dir=self.lstm_ckpt_dir, net=net, optim=optimizer)
        
        layer_freeze(net.lstm)
        
        G = Feautre_Generator(latent_size=self.latent_size, hidden_size=self.fg_hidden_size, output_size=self.lstm_hidden_size, num_classes=self.num_classes//2).to(self.device)
        g_ = optim.Adam(G.parameters(), lr=lr)
        G, _ = load_model(ckpt_dir=os.path.join(self.feature_gan_ckpt_dir,'generator'), net=G, optim=g_)

        
        for epoch in range(self.num_epoch):
            
            net.train()
            train_loss_arr = []
            train_acc_arr = []
            
            for i, (_, labels) in enumerate(train_loader):
                
                batch_size = len(labels)
                z = Variable(torch.randn(batch_size, self.latent_size).to(self.device))
                labels = labels.to(self.device)

                fake_feature = G(z,labels)
                
                outputs = net.classifier_retrain(fake_feature)
                pred = fn_pred(outputs)
                
                optimizer.zero_grad()
                
                loss = criterion(outputs, labels)
                acc = fn_acc(pred,labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss_arr+=[loss.item()]
                train_acc_arr+=[acc.item()]
            
            with torch.no_grad():
                net.eval()
                test_loss_arr = []
                test_acc_arr = []
                
                for input_value, labels in test_loader:
                    
                    input_value = input_value.to(self.device,dtype=torch.float32)
                    labels = labels.to(self.device)
                    
                    outputs = net(input_value)
                    pred = fn_pred(outputs)
                    
                    test_loss = criterion(outputs,labels)
                    test_acc = fn_acc(pred,labels)
                    
                    test_loss_arr+=[test_loss.item()]
                    test_acc_arr+=[test_acc.item()]
            
            print('Epoch [{}/{}]\ttrain_Loss: {:.4f}, train_Accuracy: {:.4f}\ttest_Loss: {:.4f}, test_Accuracy: {:.4f}' .format(epoch+1, self.num_epoch, np.mean(train_loss_arr) , np.mean(train_acc_arr), 
                                                                                                                                np.mean(test_loss_arr) , np.mean(test_acc_arr)))
            
            
            if min_train_loss > np.mean(train_loss_arr):
                min_train_loss = np.mean(train_loss_arr)
                save_model(ckpt_dir=self.lstm_retrain_ckpt_dir, net=net, optim=optimizer, epoch=epoch)
                
            writer.add_scalar('train_loss', np.mean(train_loss_arr), epoch)
            writer.add_scalar('train_acc', np.mean(train_acc_arr), epoch)
            writer.add_scalar('test_loss', np.mean(test_loss_arr), epoch)
            writer.add_scalar('test_acc', np.mean(test_acc_arr), epoch)
            
        writer.close()
