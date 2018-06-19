import numpy as np
import astropy
import scipy
import time
import torch
from astropy.table import Table
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn
import h5py as h5


#function of reading file from grams fits file
#It will give out the data tag and the data
def ss_readfits(fitsfile):
    if fitsfile[-5:] != '.fits': 
        fitsfile+='.fits'
    data=Table.read(fitsfile)
    tags=data.colnames
    return data, tags

#filter the function from the grams data
def Filter_function(COR,R,Ms,g,data):
    return np.where((data['C2O']==COR)&(data['Rin']==R)&(data['Mass']==Ms)&(data['logg']==g))[0]


'''
This is a funny function, not in use in this training script but I still put it here
This a function to switch the label as (wavelength,Teff,logg,Rin,etc...) (normally we dont put
 wavelength as a training label/parameter)
'''
def special_packing_function(index_pos):
	pre_label,pre_spectra=np.array([tau[index_pos],Teff[index_pos],logg[index_pos]\
		,CO2[index_pos],Rin[index_pos]]).T,np.array(np.log(flux[index_pos]));
	label      =np.zeros((pre_label.shape[1]+1,pre_label.shape[0]*pre_spectra.shape[1]));
	spectra    =np.zeros(pre_label.shape[0]*pre_spectra.shape[1]);
	for i in xrange(pre_label.shape[0]):
		spectra[i*pre_spectra.shape[1]:(i+1)*pre_spectra.shape[1]] =np.copy(pre_spectra[i])
		label[-1,i*pre_spectra.shape[1]:(i+1)*pre_spectra.shape[1]]=np.copy(wlen[1])
		for j in xrange(pre_label.shape[1]):
			label[j,i*pre_spectra.shape[1]:(i+1)*pre_spectra.shape[1]]=np.copy(pre_label[i][j])
	label  = np.array(label,dtype=np.float32).T;
	spectra= np.array(spectra,dtype=np.float32).T;
	return label,spectra

'''
converting the numpy ndarray to the format of the pytorch tensor array
'''
def packing_function(t_x,t_y):
	 x=torch.from_numpy(t_x)
	 y=torch.from_numpy(t_y)
	 return x.float(),y.float()

'''
A function to transate the data from grams fits file to pytorch tensor file
It will not transate all the data in the testing (by default 80% that controled by the parameter ratio)
For real world use, you  set it to 100 but i suggest 95 to 99 (use the 1~5 % monitior is that overfit)
'''
def data_to_pytorch(filepath=None,ratio=80):
	if filepath is None:
		filepath='grams_c.fits';
	print(filepath)

	#reading the data from fits file

	data,tags=ss_readfits(filepath);
	tau  =data['tau11_3']
	Mass =data['Mass']
	Teff =data['Teff']
	CO2  =data['C2O']
	logg =data['logg']
	flux =data['Fspec']
	wlen =data['Lspec']  
	Rin  =data['Rin']
	index_pos=np.where(Mass==2.0)[0]
	label,spectra=np.array([tau[index_pos],Teff[index_pos],logg[index_pos],CO2[index_pos],Rin[index_pos]]).T,np.array(np.log(flux[index_pos]));
	
	
	'''
	#rescale the label to make label to be 0.1~0.9
	The x_min is the min of all labels (Teff,logg ,etc)
	THe x_Max is the max of all labels
	'''
	x_max= np.max(label.T, axis=1)
	x_min = np.min(label.T, axis=1)
	label = ((label-x_min)*0.8/(x_max-x_min) +0.1)

    ################################################################################################################

	'''
		using the numpy random library to help to generate the random order
		the ratio is we put how many fits data as the training example (example : ratio :80  ~80% of the data will be
	training example. The rest 20% will be the testing example.

	'''
	shuffle_index=np.arange(len(label));
	np.random.shuffle(shuffle_index);
	spectra_train=np.copy(spectra[shuffle_index[0:len(label)*ratio/100]]);
	label_train  =np.copy(label[shuffle_index[np.arange(len(label)*ratio/100)]]);
	label_test   =np.copy(label[shuffle_index[len(label)*ratio/100+1:len(label)]]);
	spectra_test =np.copy(spectra[shuffle_index[len(label)*ratio/100+1:len(label)]]);

	#packing the data from np ndarray to torch data
	training_x,training_y=packing_function(label_train,spectra_train)
	testing_x,testing_y  =packing_function(label_test,spectra_test)

	return training_x,training_y,testing_x,testing_y,x_max,x_min,wlen

#####################################################################################################################
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
net=Net(training_x.numpy().shape[1],training_y.numpy().shape[1],None)
#move the network from cpu to gpu
train_function(training_x,training_y,iter_no=1000,LR=0.005,plot_i=1610)
net=net.cpu()
