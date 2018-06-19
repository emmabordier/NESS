#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:30:52 2018

@author: bordieremma
"""

import numpy as np
import astropy
import scipy
import time
import torch
from astropy.table import Table
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import h5py as h5

# Labels=names (OAGB=1 / CAGB=2 , RSG= 3 'cause Pytorch can only use int and floats)

def data_to_pytorch(filepath=None,ratio=95):
    def ss_readfits(fitsfile):
	    if fitsfile[-5:] != '.fits': 
	        fitsfile+='.fits'
	    data=Table.read(fitsfile)
	    tags=data.colnames
	    return data, tags


    def packing_function(t_x,t_y):
        x=torch.from_numpy(t_x)
        y=torch.from_numpy(t_y)
        return x.float(), y.float()
    
    if filepath is None:
        filepath='SAGE2.fits';
    print(filepath)
    
    data, tags=ss_readfits(filepath)
    
    name=data['Name']
    wlen=data['Wave']
    flux=data['Flux']
    dflux=data['Dflux']
    subtype=data['Subtype']
    
    name[name=='OAGB']=1
    name[name=='CAGB']=2
    name[name=='RSG']=3
    #print(name)
    name1=[float(i) for i in name]
    name2=np.array(name1)
    #print(name2)
            
    a=np.random.randint(0,len(flux),30)
    ind_flux=np.int64(a)
    ind_name2=np.int64(a)
    
    flux2=flux[ind_flux]
    name3=name2[ind_name2]
    
    print(flux2.shape,flux2)
    print(name3.shape,name3)
    
    
    index_pos=np.where(np.isnan(flux2)==False)[0]  #np.where((Mass==2.0)&(tau!=4.0))[0]
    print('index_pos',np.size(index_pos))
    label,spectra=np.array(name3[index_pos]),np.array(np.log(flux2[index_pos]));
                          
    x_max= np.max(label.T, axis=0)
    x_min = np.min(label.T, axis=0)
    label = ((label-x_min)*0.8/(x_max-x_min) +0.1)
    
    shuffle_index=np.arange(len(label));
    np.random.shuffle(shuffle_index);
    spectra_train=np.copy(spectra[shuffle_index[0:len(label)*ratio//100]]);
    label_train  =np.copy(label[shuffle_index[np.arange(len(label)*ratio//100)]]);
    label_test   =np.copy(label[shuffle_index[len(label)*ratio//100+1:len(label)]]);
    spectra_test =np.copy(spectra[shuffle_index[len(label)*ratio//100+1:len(label)]]);

	#packing the data from np ndarray to torch data
    training_x,training_y=packing_function(label_train,spectra_train)
    testing_x,testing_y  =packing_function(label_test,spectra_test)
    
    return training_x,training_y,testing_x,testing_y,x_max,x_min,wlen

class Net(torch.nn.Module):
	def __init__(self,input_size,output_size,ner_size):
		super(Net, self).__init__()
		self.input_layer         =torch.nn.Linear(input_size,4*input_size, bias=True)
		self.hidden_feature_1    =torch.nn.Linear(4*input_size,output_size, bias=True)
		self.hidden_feature_2    =torch.nn.Linear(output_size,output_size, bias=True)
		self.predict_layer       =torch.nn.Linear(output_size,output_size, bias=True)

	def forward(self,x):
		x=F.sigmoid(self.input_layer(x))
		x=F.sigmoid(self.hidden_feature_1(x))
		x=F.sigmoid(self.hidden_feature_2(x))

		x=self.predict_layer(x)
		return x

def file_write(txt_name,net,loss_function,tr_x,tr_y,te_x,te_y,epoch,wtime,file_w=None):
	if epoch%wtime==0 and epoch!=0:
		net.eval()
		real_pred = net(tr_x)
		real_cost = loss_function(real_pred,tr_y)
		test_pred = net(te_x)
		test_cost = loss_function(test_pred,te_y)
		training_display_cost=np.exp(np.sqrt(real_cost[0].cpu().data.numpy())/training_x.numpy().shape[0]/training_y.numpy().shape[1])-1
		testing_display_cost =np.exp(np.sqrt(test_cost[0].cpu().data.numpy())/testing_x.numpy().shape[0]/testing_y.numpy().shape[1])-1
		if not file_w:
			print(str(epoch)+',training error:'+str(training_display_cost)+',testing  error:'+str(testing_display_cost))
		else:
			file = open(txt_name,'a')
			file.write(str(epoch)+',training error:'+str(training_display_cost)+',testing  error:'+str(testing_display_cost)+'\n')
			file.close()
            
def cost_check(training_x,error):
	error=np.exp(np.sqrt(error.cpu().data.numpy()*1.0/training_x.numpy().shape[0]/training_y.numpy().shape[1]))-1
	return error

def train_function(training_x,training_y,iter_no=3000,LR=0.005,plot_i=10):

	#this a type of gradient descent method called Adam.
	#We call this optimizer , it is use for calculating the parameter error to optimize/train the function 
	optimizer=torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
	
	#this is the loss function,I am using the least sqaure fit error function in this case (without divide the number)
	loss_function=torch.nn.MSELoss(size_average=False)


	#put the testing data to the gpu
	te_x=Variable(testing_x)
	te_y=Variable(testing_y).float()

	# move the data to gpu
	tr_x=Variable(training_x)
	tr_y=Variable(training_y).float()

	for epoch in range(iter_no):
		#set the grad to be zero(no meaning)
		optimizer.zero_grad(); 
		
		# doing the predection
		prediction = net(tr_x) 
		
		#calcatlating the error 
		cost= loss_function(prediction,tr_y),
		cost[0].backward()
		
		#doing the optimization
		optimizer.step()
		
		if epoch % 2 == 0:
        # plot and show learning process
			#plt.cla()
			x=testing_x[plot_i]
			x=Variable(torch.unsqueeze(x,dim=1)).contiguous().view(1,training_x.numpy().shape[1])
			y = net(x)
			plt.ylim(np.exp(testing_y[plot_i].numpy().min())*1.0/2,np.exp(testing_y[plot_i].numpy().max())*5.0)
			plt.loglog(wlen[1],np.exp(testing_y[plot_i].numpy()) ,'o',label='real')
			plt.loglog(wlen[1],np.exp(y.data.numpy().T), 'r-', lw=5,label='predicted data')
			plt.text(1, np.exp(testing_y[plot_i].numpy().min())/1.2,'cost=%.4f' % cost_check(training_x,cost[0])[0], fontdict={'size': 18, 'color':  'red'})
			plt.text(1, np.exp(testing_y[plot_i].numpy().min())*5.0,'loop='+str(epoch), fontdict={'size': 18, 'color':  'red'})	
			plt.legend(loc="lower right")
			plt.xlabel("wavelength(um)")
			plt.ylabel("flux")		
			#plt.pause(0.01)
			plt.savefig(str(epoch)+".png")
			#plt.savefig("-1.png")
			plt.close()
            
       

#get the training and testing data
training_x,training_y,testing_x,testing_y,x_max,x_min,wlen=data_to_pytorch(filepath=None,ratio=95);
start_time=time.time()
net=Net(training_x.numpy().shape[0],training_y.numpy().shape[0],None)
#move the network from cpu to gpu
train_function(training_x,training_y,iter_no=1,LR=0.005,plot_i=1610)
#net=net.cpu()


                          
                          
                          
                          
                          
                          
                          
                          