import numpy as np
import torch
from torch.autograd import Variable
from VAE import autoencoder,predict,valloader,show_encode,rand_predict

model=autoencoder().cuda()
model.load_state_dict(torch.load('./weights/Variational_autoencoder_min.pth'))
predict(valloader,model)
#rand_predict(model)
#show_encode(valloader,model)