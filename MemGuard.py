import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from numpy import genfromtxt
import copy
import numpy as np
from train import *
from model import *
from dataset import *

def MemGuard(args, in_data, out_data, defender, softmax, lr = 0.3, max_iter = 99):
    device = args.device
    # running memguard adding noise
    mem_loder = [out_data, in_data]
    perturbation = [torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)]

    defender = defender.to(device)
    #max_iter = 99
    calculator = [0, 0, 0, 0]
    iter_list = [[],[]]
    for j in range(len(mem_loder)):
        mem_data, category_label, mem_label = mem_loder[j]
        assert len(mem_data.shape) == 2 #check
        iter_list[j] = [0 for _ in range(len(mem_label))]
        for pointer in range(len(mem_label)):
            # set model and loss
            criterion = MemGuardLoss()
            defender.eval()
            #calculate logits
            logits = torch.index_select(mem_data, dim = 0, index = torch.tensor([pointer])).to(device)
            # noise e
            e = nn.Parameter(data=torch.zeros_like(logits),requires_grad=True).to(device)
            # l=argmax(z)
            l = torch.topk(logits, 1)[1].item()
            # h(softmax(z))
            attacker_logits = defender(softmax(logits))
            # set learning rate
            #lr = 0.1
            if abs(attacker_logits.item()) > 1e-5:
                while True:
                    e_p = e.clone().detach()
                    e = nn.Parameter(data=torch.zeros_like(logits), requires_grad=True).to(device)
                    i = 0
                    while i < max_iter and (l != torch.topk(logits + e, 1)[1].item() or attacker_logits.item() *
                                            defender(softmax(logits + e)).item() > 0):
                        defender.zero_grad()
                        loss = criterion(error=e, logits=logits, model=defender, l=l, softmax_operation=softmax)
                        e.retain_grad()
                        loss.backward()
                        e_grad = e.grad.detach()
                        e_grad = e_grad / (torch.norm(e_grad, p=2, dim=1) + 1e-7)
                        e = nn.Parameter(data=e - lr * e_grad, requires_grad=True).to(device)
                        i = i + 1
                        iter_list[j][pointer] += 1
                    if torch.isnan(e).any().item():
                        if torch.norm(e_p, p=2).item() < 1e-5:
                            calculator[-1] += 1
                        break
                    if l != torch.topk(logits + e, 1)[1].item():
                        if torch.norm(e_p, p=2).item() < 1e-5:
                            calculator[0] += 1
                        break
                    if attacker_logits.item() * defender(softmax(logits + e)).item() > 0:
                        if torch.norm(e_p, p=2).item() < 1e-5:
                            calculator[1] += 1
                        break
                    if criterion.c3 >= 100000: #100000
                        e_p = e.clone().detach()
                        calculator[2] += 1
                    criterion.c3 = criterion.c3 * 10
            else:
                e_p = nn.Parameter(torch.zeros_like(logits)).to(device)
            with torch.no_grad():
                perturbation[j] = torch.cat((perturbation[j], e_p.cpu().detach()), 0)
    with open('./iteration/' + str(args.sign) + 'iter_.npy', 'wb') as f:
        np.save(f, np.array(iter_list, dtype=object))
    try:
        print(calculator)
    except:
        pass
    return perturbation


class MemGuardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = 1
        self.c2 = 10.
        self.c3 = 0.1

    def forward(self, error, logits, model, l, softmax_operation=nn.Softmax(dim=1)):
        # softx(z+e))
        soft_ze = softmax_operation(logits + error)  # soft_ze = torch.nn.functional.softmax(logits+error, 1)

        # h(softx(z+e))
        new_logits = model(soft_ze)

        # |-h(softmax(z+e))|
        L1 = torch.abs(new_logits)

        # argmax(z+e) calculate l_p (logits+error).shape == torch.Size([1, 8])
        top_2_index = torch.topk(logits + error, k=2)[1][0] #top_2_index.shape == torch.Size([2])
        if top_2_index[0].item() == l:
            l_p = top_2_index[1].item()
        else:
            l_p = top_2_index[0].item()
        L2 = torch.nn.functional.relu(-1 * (logits + error)[0][l] + (logits + error)[0][l_p])

        # softx(z))
        soft_z = softmax_operation(logits)  # soft_z = torch.nn.functional.softmax(logits, 1)
        L3 = torch.sum(torch.abs(soft_ze - soft_z))

        return self.c1 * L1 + self.c2 * L2 + self.c3 * L3

class SecureML_softmax(torch.nn.Module):
    def __init__(self):
        super(SecureML_softmax, self).__init__()

    def forward(self, input):
        relu_input = nn.ReLU()(input)
        s = torch.sum(relu_input,dim=1,keepdim=True)
        L = 1/relu_input.shape[1]
        return torch.where(s>0,(1/s) * relu_input,L)

class Piranha_softmax(torch.nn.Module):
    def __init__(self):
        super(Piranha_softmax, self).__init__()

    def forward(self, input):
        input = input - torch.max(input, dim=1, keepdim=True)[0]
        input = torch.where(input<-2,0,input*0.5+1)
        return input/torch.sum(input,dim=1,keepdim=True)

class Abdelrahaman_softmax(torch.autograd.Function):
    def __init__(self):
        super(Abdelrahaman_softmax, self).__init__()

    @staticmethod
    def forward(ctx, input):
        p_1045 = torch.tensor([math.log(2) ** i / math.factorial(i) for i in range(100)])
        log2_e = torch.tensor(math.log(math.exp(1), 2))

        input = input - torch.mean(input,1).unsqueeze_(1)

        abs_input = torch.absolute(input)
        abs_input = log2_e * abs_input

        decimal = abs_input.frac()
        integer = (abs_input - decimal).clamp_(min=-150, max=125)

        input_e = torch.zeros_like(decimal)
        for i in range(100):
          input_e += p_1045[i]*(decimal**i)
        input_e = torch.pow(2, integer) * input_e
        input_e = torch.where(input<0,1/input_e,input_e)
        s = torch.sum(input_e,dim=1,keepdim=True)
        output = input_e/s
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        data_len, feature_len = grad_output.shape
        grad_o = grad_output.clone().unsqueeze_(1).expand((-1,feature_len, -1))

        eye = torch.eye(feature_len).to('cuda')
        eye = eye.unsqueeze_(0).expand((data_len,-1, -1))

        output, = ctx.saved_tensors
        output = output.unsqueeze_(1).expand((-1,feature_len, -1))

        o = grad_o * torch.transpose(output, 1, 2) * (eye-output)
        o = o.sum(dim=2)
        return o

class AS19_softmax(torch.nn.Module):
    def __init__(self):
        super(AS19_softmax, self).__init__()
        self.softmax = Abdelrahaman_softmax()

    def forward(self, input):
        return self.softmax.apply(input)

class crypten(torch.autograd.Function):
    def __init__(self):
        super(crypten, self)

    @staticmethod
    def forward(ctx, input):
        input = input - torch.max(input,1).values.unsqueeze(1)
        input_e = (1+input/(2**9))**(2**9)
        s = torch.sum(input_e,dim=1,keepdim=True)
        output = input_e/s
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_o = grad_output.clone()
        o = torch.zeros_like(grad_o)
        data_len, feature_len = grad_o.shape

        eye = torch.eye(feature_len).to('cuda')

        output, = ctx.saved_tensors
        for k in range(data_len):
          p = output[k].expand((feature_len, -1))
          g = grad_o[k].expand((feature_len, -1))
          all = g*p.t()*(eye - p)
          o[k] = all.sum(dim=1)
          '''
          for i in range(feature_len):
            for j in range(feature_len):
              o[k][i] += grad_o[k][j]*output[k][i]*(int(i==j) - output[k][j])
          '''
        return o

class CrypTen_softmax(torch.nn.Module):
    def __init__(self):
        super(CrypTen_softmax, self).__init__()
        self.softmax = crypten()

    def forward(self, input):
        return self.softmax.apply(input)