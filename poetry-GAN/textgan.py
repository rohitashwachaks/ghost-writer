#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai import *
from fastai.text import *
from fastai.imports import *


import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader

import pandas as pd


# In[3]:


from lang_model import *


# In[4]:


# get_ipython().system('conda list fastai')


# In[5]:


path = Path('../data/')
path


# In[6]:


from sklearn.model_selection import train_test_split

def load_data(df: pd.DataFrame):
    train_sentences, valid_sentences = train_test_split(sentences, test_size= 0.1, random_state = 121)

    data_lm = TextLMDataBunch.from_df(path = path, train_df= sentences, valid_df= valid_sentences, text_cols= "0")
    return data_lm


# # Read CSV

# In[7]:


sentences = pd.read_csv("../data/lyrics_preprocessed.csv")
sentences


# In[10]:


data_lm = load_data(sentences)


# In[12]:


trn_dl = data_lm.train_dl
val_dl = data_lm.valid_dl


# In[13]:


#export
def bn_drop_lin(n_in, n_out, bn=True, initrange=0.01,p=0, bias=True, actn=nn.LeakyReLU(inplace=True)):
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    linear = nn.Linear(n_in, n_out, bias=bias)
    if initrange:linear.weight.data.uniform_(-initrange, initrange)
    if bias: linear.bias.data.zero_()
    layers.append(linear)
    if actn is not None: layers.append(actn)
    return layers


# In[14]:


#export
def bn_drop_lin(n_in, n_out, bn=True, initrange=0.01,p=0, bias=True, actn=nn.LeakyReLU(inplace=True)):
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    linear = nn.Linear(n_in, n_out, bias=bias)
    if initrange:linear.weight.data.uniform_(-initrange, initrange)
    if bias: linear.bias.data.zero_()
    layers.append(linear)
    if actn is not None: layers.append(actn)
    return layers


# In[15]:


from fastai.text.learner import language_model_learner
from fastai.text.models import TransformerXL
learn = language_model_learner(data_lm, arch=TransformerXL)
# learn.load('lyrics_fine_tuned_novel')


# In[16]:


encoder = deepcopy(learn.model[0])
encoder


# In[17]:


x, y = next(iter(trn_dl))
x.size(), y.size()


# In[18]:


outs = encoder(x)


# In[19]:


outs[-1][-1].size()


# In[20]:


[out.size() for out in outs[-1]]


# In[21]:


generator = deepcopy(learn.model) 


# In[22]:


generator.load_state_dict(learn.model.state_dict())


# In[23]:


#export
class TextDicriminator(nn.Module):
    def __init__(self,encoder, nh, bn_final=True):
        super().__init__()
        #encoder
        self.encoder = encoder
        #classifier
        layers = []
        layers+=bn_drop_lin(nh*3,nh,bias=False)
        layers += bn_drop_lin(nh,nh,p=0.25)
        layers+=bn_drop_lin(nh,1,p=0.15,actn=nn.Sigmoid())
        if bn_final: layers += [nn.BatchNorm1d(1)]
        self.layers = nn.Sequential(*layers)
    
    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(0,2,1), (1,)).view(bs,-1)
    
    def forward(self, inp,y=None):
        raw_outputs, outputs = self.encoder(inp)
        output = outputs[-1]
        bs,sl,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[:,-1], mxpool, avgpool], 1)
        out = self.layers(x)
        return out


# In[24]:


disc = TextDicriminator(encoder,400)#.cuda()


# In[25]:


optimizerD = optim.Adam(disc.parameters(), lr = 3e-4)
optimizerG = optim.Adam(generator.parameters(), lr = 3e-3, betas=(0.7, 0.8))


# In[26]:


#export
def seq_gumbel_softmax(input):
    samples = []
    bs,sl,nc = input.size()
    for i in range(sl): 
        z = F.gumbel_softmax(input[:,i,:])
        samples.append(torch.multinomial(z,1))
    samples = torch.stack(samples).transpose(1,0).squeeze(2) 
    return samples


# In[27]:


#export
from tqdm import tqdm


# In[28]:


#export
def reinforce_loss(input,sample,reward):
    loss=0
    bs,sl = sample.size()
    for i in range(sl):
        loss += -input[:,i,sample[:,i]] * reward
    return loss/sl


# In[29]:


#export
def step_gen(ds,gen,disc,optG,crit=None):
    gen.train(); disc.train()
    x,y = ds
    bs,sl = x.size()
    fake,_,_ = gen(x)
    gen.zero_grad()
    fake_sample =seq_gumbel_softmax(fake)
    with torch.no_grad():
        gen_loss = reward = disc(fake_sample)
        if crit: gen_loss = crit(fake,fake_sample,reward.squeeze(1))
        gen_loss = gen_loss.mean()
    gen_loss.requires_grad_(True)
    gen_loss.backward()
    optG.step()
    return gen_loss.data.item()


# In[30]:


#export
def step_disc(ds,gen,disc,optD,d_iters):
    for j in range(d_iters):
        gen.eval(); disc.train()
        with torch.no_grad():
            fake,_,_ = gen(x)
            fake_sample = seq_gumbel_softmax(fake)
        disc.zero_grad()
        fake_loss = disc(fake_sample)
        real_loss = disc(y.view(bs,sl))
        disc_loss = (fake_loss-real_loss).mean(0)
        disc_loss.backward()
        optimizerD.step()
    return disc_loss.data.item()


# In[31]:


#export
def evaluate(ds,gen,disc,crit=None):
    with torch.no_grad():
        x, y = ds
        bs,sl = x.size()
        fake,_,_ = gen(x)
        fake_sample =seq_gumbel_softmax(fake)
        gen_loss = reward = disc(fake_sample)
        if crit: gen_loss = crit(fake,fake_sample,reward.squeeze(1))
        gen_loss = gen_loss.mean()
        fake_sample = seq_gumbel_softmax(fake)
        fake_loss = disc(fake_sample).mean(0).view(1).data.item()
        real_loss = disc(y.view(bs,sl)).mean(0).view(1).data.item()
        disc_loss = (fake_loss-real_loss).mean(0).view(1).data.item()
    return fake,gen_loss,disc_loss,fake_loss


# In[32]:


#export
def train(gen, disc, epochs, trn_dl, val_dl, optimizerD, optimizerG, crit=None,first=True):
    gen_iterations = 0
    
    for epoch in range(epochs):
        
        gen.train(); disc.train()
        n = len(trn_dl)
        #train loop
        with tqdm(total=n) as pbar:
            for i, ds in enumerate(trn_dl):
                gen_loss = step_gen(ds,gen,disc,optimizerG,crit)
                gen_iterations += 1
                d_iters = 3
                disc_loss = step_disc(ds,gen,disc,optimizerD,d_iters)
                pbar.update()
        print(f'Epoch {epoch}:')
        print('Train Loss:')
        print(f'Loss_D {disc_loss}; Loss_G {gen_loss} Ppx {torch.exp(lm_loss(fake,y))}')
        print(f'D_real {real_loss}; Loss_D_fake {fake_loss}')
        
        disc.eval(), gen.eval()
        with tqdm(total=len(val_dl)) as pbar:
            for i, ds in enumerate(val_dl):
                fake,gen_loss,disc_loss,fake_loss = evaluate(ds,gen,disc,crit)
                pbar.update()
        print('Valid Loss:')
        print(f'Loss_D {disc_loss}; Loss_G {gen_loss} Ppx {torch.exp(lm_loss(fake,ds[-1]))}')
        print(f'D_real {real_loss}; Loss_D_fake {fake_loss}')


# In[33]:


#export
nh = {'AWD':400,'XL':410}
crits={'gumbel':None,'reinforce':reinforce_loss}

#train a language model with gan objective
def run(path,filename,pretrained,model,crit=None,preds=True,epochs=6):
    #load data after running preprocess
    print(f'loading data from {path}/{filename};')
    data_lm = load_data(path, filename)
    trn_dl = data_lm.train_dl
    val_dl = data_lm.valid_dl
    
    #select encoder for model
    print(f'training text gan model {model}; pretrained from {pretrained};')
    learn = language_model_learner(data_lm, arch=models[model])
    learn.load(pretrained)
    encoder = deepcopy(learn.model[0])
    
    generator = deepcopy(learn.model)
    generator.load_state_dict(learn.model.state_dict())
    disc = TextDicriminator(encoder,nh[model]).cuda()
    
    disc.train()
    generator.train()
    
    #create optimizers
    optimizerD = optim.Adam(disc.parameters(), lr = 3e-4)
    optimizerG = optim.Adam(generator.parameters(), lr = 3e-3, betas=(0.7, 0.8))
    
    print(f'training for {epochs} epochs')
    train(generator, disc, epochs, trn_dl, val_dl, optimizerD, optimizerG, first=False)
    
    #save model
    learn.model.load_state_dict(generator.state_dict())
    print(f'saving model to {path}/{filename}_{model}_gan_{crit}')
    learn.save(filename+'_'+model+'_gan_'+crit)
    
    #generate output from validation set
    if preds:
        print(f'generating predictions and saving to {path}/{filename}_{model}_preds.txt;')
        get_valid_preds(learn,data_lm,filename+'_'+model+'_preds.txt')


# In[ ]:





# In[34]:


#export
if __name__ == '__main__':
    fire.Fire(run)