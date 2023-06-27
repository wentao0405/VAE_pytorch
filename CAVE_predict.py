import matplotlib as plt
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torchvision.utils import save_image
import qqdm
import numpy as np
from torch.autograd import Variable
from CVAE import autoencoder,predict_from_text,valloader,show_encode,predict_from_img
text=[0,1,2,3,4,5,6,7,8,9]
model=autoencoder().cuda()
model.load_state_dict(torch.load('./weights/Contro_Variational_autoencoder_min.pth'))
#predict_from_img(valloader,model)
predict_from_text(text,model)

#show_encode(valloader,model)