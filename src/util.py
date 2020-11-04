import os
import torch
import torch.nn.functional as F
import numpy as np
import itertools
import matplotlib.pyplot as plt

# open source
# https://deeplizard.com/learn/video/0LhiS6yu2qQ

def plot_confusion_matrix(cm, classes, save_dir, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.cla()
    plt.clf()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_dir)
    
def save_model(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    torch.save({'net':net.state_dict(), 'optim': optim.state_dict()},
                '%s/model_epoch_%d.pth'%(ckpt_dir,epoch))

def load_model(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
        
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()
    
    dict_model = torch.load('%s/%s' %(ckpt_dir, ckpt_lst[-1]))
    
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    return net, optim

def sample_z(batch_size, d_noise, device):
    return torch.randn(batch_size, d_noise, device = device)


def layer_freeze(layer):
    for param in layer.parameters():
        param.requires_grad = False
        

# source
# https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
class CB_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        samples_per_cls: A python list of size [no_of_classes].
        no_of_classes: total number of classes. int
        loss_type: string. One of "sigmoid", "focal", "softmax".
        beta: float. Hyperparameter for Class balanced loss.
        gamma: float. Hyperparameter for Focal loss.
        Returns:
        cb_loss: A float tensor representing class balanced loss
        """
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        labels_one_hot = one_hot_embedding(labels, no_of_classes).float()

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)

        if loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
        elif loss_type == "softmax":
            m = torch.nn.Softmax()
            logits = m(logits)
            cb_loss = F.binary_cross_entropy(input = logits, target = labels_one_hot, weight = weights)
        return cb_loss

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred = ctx.saved_tensors
        grad_input = torch.neg(2.0 * (yy_pred - ctx.y))
        return grad_input, grad_output



# source
# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/25
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]



# source
# https://github.com/hanyoseob/pytorch-StarGAN/blob/master/utils.py
class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)
        print('\n\n')
        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)
        print('\n\n')

