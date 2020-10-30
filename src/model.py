import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class RNN_MODEL(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, sequence_length, num_lstm, num_classes, dropout, device):
        super(RNN_MODEL, self).__init__()
        self.hidden_size = hidden_size
        self.num_lstm = num_lstm
        self.lstm = nn.LSTM(input_size, hidden_size, num_lstm, dropout=dropout, batch_first=True)
        self.classifier_1 = nn.Sequential(
            nn.Linear(self.hidden_size, num_classes)
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(self.hidden_size,50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )
        self.device = device
    
    def forward(self, x):

        
        h0 = torch.zeros(self.num_lstm, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_lstm, x.size(0), self.hidden_size).to(self.device)


        out, (ht, ct) = self.lstm(x, (h0, c0))

        if self.hidden_size >= 100:
            out = self.classifier_2(ht[-1])
        else:
            out = self.classifier_1(ht[-1])
        return out

    def feature_extraction(self, x):
        h0 = torch.zeros(self.num_lstm, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_lstm, x.size(0), self.hidden_size).to(self.device)

        out, (ht, ct) = self.lstm(x, (h0, c0))
        
        return ht[-1]

    def classifier_retrain(self, x):
        return self.classifier_2(x)

class Feautre_Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_classes):
        super(Feautre_Generator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, latent_size)
        self.generator = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
            
    def forward(self, x, labels):
        x = torch.mul(self.label_emb(labels), x)
        return self.generator(x)

class Feature_Discriminator(nn.Module):
    def __init__(self, feature_size, hidden_size, num_classes):
        super(Feature_Discriminator, self).__init__()
        self.discriminator_binary = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.discriminator_class = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.discriminator_binary(x), self.discriminator_class(x)

