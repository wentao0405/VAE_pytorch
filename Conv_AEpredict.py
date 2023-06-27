import matplotlib as plt
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torchvision.utils import save_image
import qqdm
import numpy as np
from torch.autograd import Variable
from Conv_autoencoder import autoencoder,predict,valloader,show_encode

model=autoencoder().cuda()
model.load_state_dict(torch.load('./weights/Conv_autoencoder_end.pth'))
predict(valloader,model)
#show_encode(valloader,model)