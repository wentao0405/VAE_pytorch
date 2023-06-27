import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torchvision.utils import save_image
import qqdm
import numpy as np
from torch.autograd import Variable

num_epochs=300
Batchsize=64
validation_split=0.2
dataset=torchvision.datasets.MNIST('./',True,transform=torchvision.transforms.ToTensor(),download=True)
dataset_size=len(dataset)
indices=list(range(dataset_size))
split=int(np.floor(validation_split*dataset_size))
train_indices,val_indices=indices[split:],indices[:split]
train_sampler=Data.SubsetRandomSampler(train_indices)
valid_sampler=Data.SubsetRandomSampler(val_indices)

trainloader=Data.DataLoader(dataset=dataset,shuffle=False,batch_size=Batchsize,sampler=train_sampler)
valloader=Data.DataLoader(dataset=dataset,shuffle=False,batch_size=Batchsize,sampler=valid_sampler)
learning_rate=1e-3

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
def encode_to_img(x):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 16, 2)
    return x

class autoencoder(nn.Module):
    def __init__(self,inference=False):
        super(autoencoder,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(1,16,3,stride=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(16,8,3,stride=2,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=1)
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(8,16,3,stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,5,stride=3,padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8,1,2,stride=2,padding=1),
            nn.Tanh()
        )
        self.latent_dim=64
        self.img_size = 28
        self.decode_latent_dim=32
        self.mean=nn.Sequential(
            nn.Linear(in_features=self.latent_dim,out_features=self.latent_dim*2),
            nn.ReLU(True),
            nn.Linear(in_features=self.latent_dim*2,out_features=self.decode_latent_dim,bias=False)
        )
        self.log_var=nn.Sequential(
            nn.Linear(in_features=self.latent_dim,out_features=self.latent_dim*2),
            nn.ReLU(True),
            nn.Linear(in_features=self.latent_dim*2,out_features=self.decode_latent_dim,bias=False),
        )
        self.decoder_input_size=[2,8]
        #讲latent张量映射到decoder的输入维度
        self.decoder_dense=nn.Sequential(
            nn.Linear(in_features=self.latent_dim,out_features=self.decode_latent_dim),
            nn.ReLU(True),
        )
        self.label_dim = 128

        self.inference = inference
        self.label_embedding = nn.Embedding(10, self.label_dim)
        self.label_embedding.requires_grad = True
        self.encoder_y_dense = nn.Sequential(
            nn.Linear(in_features=self.label_dim, out_features=self.img_size * self.img_size),
            nn.ReLU(True)
        )
        self.decoder_y_dense = nn.Sequential(
            nn.Linear(in_features=128,
                      out_features=self.decode_latent_dim),
            nn.ReLU(True)
        )

    def sample_latent(self,bs,image):

        z,mean,log_var=None,None,None
        x = image
        x = self.encoder(x)
        x = x.view(x.size(0), 1, 64)
        mean = self.mean(x)
        log_var = self.log_var(x)
        epsilon = torch.normal(mean=0, std=1, size=(x.size(0), 1, self.decode_latent_dim)).cuda()
        z = mean + torch.exp(0.5 * log_var) * epsilon

        return z,mean,log_var

    def call(self,inputs):
        bs,image,label=inputs[0],inputs[1],inputs[2]
        label_emb=self.label_embedding(label)
        label_emb=torch.reshape(label_emb,[-1,1,self.label_dim])
        if not self.inference:
            encoder_y=self.encoder_y_dense(label_emb)
            encoder_y=torch.reshape(encoder_y,[-1,1,self.img_size,self.img_size])
            image=torch.cat([encoder_y,image],dim=-1)
        else:
            image = torch.normal( mean=0,std=1,size=(label_emb.size(0), 1, 28, 28)).cuda()
            encoder_y = self.encoder_y_dense(label_emb)
            encoder_y = torch.reshape(encoder_y, [-1, 1, self.img_size, self.img_size])
            image = torch.cat([encoder_y,encoder_y*1.3],dim=-1)
        z,z_mean,z_log_var=self.sample_latent(bs,image)
        decoder_y=self.decoder_y_dense(label_emb)
        latent=torch.cat((z,decoder_y),dim=-1)
        latent=self.decoder_dense(latent)
        latent=torch.reshape(latent,[-1,8,2,2])

        #通过latent特征重建出图像
        reconstruct_img=self.decoder(latent)
        return reconstruct_img,z_mean,z_log_var

    def compute_loss(self,reconstruct_img,z_mean,z_log_var,img ):
        img=img.resize(img.size(0),img.size(1),reconstruct_img.size(2),reconstruct_img.size(3))
        l2_loss=torch.nn.functional.mse_loss(reconstruct_img,img)
        kl_loss=-0.5*(1+z_log_var-torch.square(z_mean)-torch.exp(z_log_var))
        kl_loss=torch.mean(torch.sum(kl_loss,dim=1))
        total_loss=kl_loss+l2_loss
        return total_loss

    def forward(self,x,label,inference):
        self.inference=inference
        image=x
        if not self.inference :
            bs=x.shape[0]
        else:
            bs=None
        reconstruct_img,z_mean,z_log_var=self.call((bs,x,label))
        if not self.inference:
            loss=self.compute_loss(reconstruct_img,z_mean,z_log_var,image)
        else :
            loss=None
        return reconstruct_img,loss


    def encode(self,x):
        x=self.encoder(x)
        return x

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
def train(num_epochs):
    loss_list=[]
    min_loss=100
    for epoch in range(num_epochs):
        total_loss = 0
        for data in trainloader:
            img, label= data
            img = Variable(img).cuda()
            label=Variable(label).cuda()
            # ===================forward=====================
            output ,loss= model(img,label,False)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, total_loss))
        loss_list.append(total_loss)
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch))
            if total_loss < min_loss:
                min_loss = total_loss
                torch.save(model.state_dict(), './weights/Contro_Variational_autoencoder_min.pth')
    torch.save(model.state_dict(), './weights/Contro_Variational_autoencoder_end.pth')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(num_epochs),np.array(loss_list))
    plt.savefig('./loss.png')


def predict_from_img(val_dataloder,model):
    total_loss=0
    img,output=None,None
    for data in val_dataloder:
        img,label=data
        img = Variable(img).cuda()
        label=Variable(label).cuda()
        # ===================forward=====================
        output ,loss= model(img,label,False)
        total_loss += loss.data
        # ===================log========================
    print('loss:{:.4f}'
          .format( total_loss))
    #print(output[0])
    pic1=to_img(img.cpu().data)
    pic2=to_img(output.cpu().data)
    n=10
    plt.figure(figsize=(20,4))
    for i in range(1,10):
        ax=plt.subplot(2,n,i)
        plt.imshow(pic1[i].reshape(28,28))
        plt.axis('off')
        ax=plt.subplot(2,n,n+i)
        plt.imshow(pic2[i].reshape(28, 28))
        plt.axis('off')

    plt.show()

def predict_from_text(text,model):
    img=None
    plt.figure()
    n = 10
    label = torch.tensor(text).cuda()
    for i in range(n):
        output,loss=model(img,label,True)
        # label = Variable(label).cuda()
        pic=to_img(output.cpu().data)
        for j in range(10):
            ax = plt.subplot(n, 10, n*i+j+1)
            plt.imshow(pic[j].reshape(28, 28))
            plt.axis('off')



    plt.show()






def show_encode(dataloder,model):
    total_loss = 0
    img, output = None, None
    for data in dataloder:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model.encode(img)
    #print(output.cpu().data.reshape([32,1,16,2]).shape)
    print(output.cpu().data[0])
    pic1 = to_img(img.cpu().data)
    pic2 = encode_to_img(output.cpu().data)
    print(pic2[0])
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, 10):
        ax = plt.subplot(2, n, i)
        plt.imshow(pic1[i].reshape(28, 28))
        plt.axis('off')
        ax = plt.subplot(2, n, n + i)
        plt.imshow(pic2[i].reshape(16, 2))
        plt.axis('off')

    plt.show()


if __name__=='__main__':
    train(num_epochs)


