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


'''
This is a simple deep learning algithom for the fitting the spectrum from GRAMS model, this is 
the testing version.
This library were using Pytorch and cuda gpu speed up.



#========================================IMPORTANT POINT==========================================#
FOR NON-TEST PLS USE THE ratio=95~99 ( must be int) to train the data
FOR NEW DATA, PLS RETRAIN THE MODEL ONCE!

WARNING: EVEN I SIMPLIZE THE NETWORK, CPU TRAINING IS STILL VERY LOW!!!!
#=======================================TRAINING TIME ============================================#

It depends on the iter_no and the final_error value you set, for the final_error value setted as 8.8e-6 and iter_no:350000
It finishs at about 2hours~4hours by using the cuda gpu speed up

DONT USE TOO LOW final error ! IT WILL NOT COVERAGE!

TRAINING SPEC:
SYSTEM: Ubuntu 16.04 LTS
CPU   : AMD Ryzen 1600
RAM   : 16G DDR4 2133 Mhz
GPU   : NVIDIA 1050Ti 4GB

#==================================================================================================#
'''

#filter the function from the grams data
def Filter_function(COR,R,Ms,g,data):
    return np.where((data['C2O']==COR)&(data['Rin']==R)&(data['Mass']==Ms)&(data['logg']==g))[0]

'''
A function to transate the data from grams fits file to pytorch tensor file
It will not transate all the data in the testing (by default 80% that controled by the parameter ratio)
For real world use, you  set it to 100 but i suggest 95 to 99 (use the 1~5 % monitior is that overfit)
'''
def data_to_pytorch(filepath=None,ratio=80):

	'''
	function of reading file from grams fits file
	It will give out the data tag and the data
	'''
	def ss_readfits(fitsfile):
	    if fitsfile[-5:] != '.fits': 
	        fitsfile+='.fits'
	    data=Table.read(fitsfile)
	    tags=data.colnames
	    return data, tags


	'''
	converting the numpy ndarray to the format of the pytorch tensor array
	'''
	def packing_function(t_x,t_y):
		 x=torch.from_numpy(t_x)
		 y=torch.from_numpy(t_y)
		 return x.float(),y.float()


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
	Tin  =data['Tin']
	#lock the data to Mass=2Ms
	index_pos=np.where(Mass>=2.0)[0]
	label,spectra=np.array([tau[index_pos],Teff[index_pos],logg[index_pos],CO2[index_pos],Rin[index_pos]]).T,np.array(np.log(flux[index_pos]));
	Tin=np.expand_dims(np.log(Tin[index_pos])/1.0,axis=1)
	spectra=Tin
	
	'''
	#rescale the label to make label to be 0.1~0.9
	The x_min is the min of all labels (Teff,logg ,etc)
	THe x_Max is the max of all labels
	'''
	x_max= np.max(label.T, axis=1)
	x_min = np.min(label.T, axis=1)
	label = ((label-x_min)*0.8/(x_max-x_min) +0.1)
	#index_train=np.where(Teff[index_pos]!=2800.0)[0]
	#index_test=np.where(Teff[index_pos]==2800.0)[0]


    ################################################################################################################

	'''
		using the numpy random library to help to generate the random order
		the ratio is we put how many fits data as the training example (example : ratio :80  ~80% of the data will be
	training example. The rest 20% will be the testing example.

	'''
	#spectra_train=np.copy(spectra[index_train]);
	#label_train  =np.copy(label[index_train]);
	#label_test   =np.copy(label[index_test]);
	#spectra_test =np.copy(spectra[index_test]);
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

# The nerual network setup
# All of the network have this structure:
''' 
Clas Network_name(troch.nn.Module):
	def __init__(self,.....):
		super(Network_name, self).__init__()    
		self.input_layer         =torch.nn.Linear(input_size,....)
		#===================THE HIDDEN LAYER=====================#
		self.output_layer       =torch.nn.Linear(....,output_size)

	def forward(self,x):
		x=F.activefunction(self.input_layer(x))
		x=F.activefunction(self.hidden_layer1(x))
		#===================THE HIDDEN LAYER=====================#
		x=self.output_layer(x)
		return x
'''
#The whole __init__ is the network structure setup
#THe forward is setting the active function
class Net(torch.nn.Module):
	def __init__(self,input_size,output_size,ner_size):
		# torch.nn.Linear is the Linear network layer
		super(Net, self).__init__()
		self.input_layer         =torch.nn.Linear(input_size,4*input_size, bias=True)
		self.hidden_feature_1    =torch.nn.Linear(4*input_size,output_size, bias=True)
		self.predict_layer       =torch.nn.Linear(output_size,output_size, bias=True)

	def forward(self,x):
		# F.sigmoid is a type of active function
		x=F.sigmoid(self.input_layer(x))
		x=F.sigmoid(self.hidden_feature_1(x))
		x=self.predict_layer(x)
		return x

#this is a small function to told the user the progress of the training
#only two mode:1. printing the progress in every n loops(I call it wtime in this function) loop or 2. writing it to the txt file
def file_write(txt_name,net,loss_function,tr_x,tr_y,te_x,te_y,epoch,wtime,file_w=None):
	if epoch%wtime==0 and epoch!=0:
		net.eval()
		real_pred = net(tr_x)
		real_cost = loss_function(real_pred,tr_y)
		test_pred = net(te_x)
		test_cost = loss_function(test_pred,te_y)
		training_display_cost=np.exp(np.sqrt(real_cost[0].cpu().data.numpy()/training_x.numpy().shape[0]/training_y.numpy().shape[1]))-1
		testing_display_cost =np.exp(np.sqrt(test_cost[0].cpu().data.numpy()/testing_x.numpy().shape[0]/testing_y.numpy().shape[1]))-1
		if not file_w:
			print(str(epoch)+',training error:'+str(training_display_cost)+',testing  error:'+str(testing_display_cost))
		else:
			file = open(txt_name,'a')
			file.write(str(epoch)+',training error:'+str(training_display_cost)+',testing  error:'+str(testing_display_cost)+'\n')
			file.close()


#this is use for the training function to check the error is low euongh for output the data
def cost_check(training_x,error):
	error=np.exp(np.sqrt(error.cpu().data.numpy()*1.0/training_x.numpy().shape[0]/training_y.numpy().shape[1]))-1
	return error

#training function
#iter_no is the training loops
#LR is learning rate
def train_function(training_x,training_y,final_error=4e-6,iter_no=100000,LR=0.005,mini_batch=None,GPU=True):

	def var_output(v_x,v_y,GPU=GPU):
		if GPU==True:
			if torch.cuda.device_count()>0:
				x=Variable(v_x).cuda()
				y=Variable(v_y).float().cuda()
			else:
				raise AttributeError("no GPU found!")
		else:
			x=Variable(v_x)
			y=Variable(v_y).float()
		return x,y

	if GPU==True:
		if torch.cuda.device_count()>0:
		#move the network from cpu to gpu
			net.cuda()
		else:
			raise AttributeError("no GPU found!")


	#this a type of gradient descent method called Adam.
	#We call this optimizer , it is use for calculating the parameter error to optimize/train the function 
	optimizer=torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
	
	#this is the loss function,I am using the least sqaure fit error function in this case (without divide the number)
	loss_function=torch.nn.MSELoss(size_average=False)

	# This is the mini batch training process set up
	# It is use the radomize the training data and put some of them to the train
	# By default, it is closed, you can switch it on by change mini_batch=None ->mini_batch=True
	# BE CAREFUL IT WAS VERY LOW (20% OF NORMAL SPEED) 
	Batch_size=training_x.numpy().shape[0]/5
	train_data_set=Data.TensorDataset(data_tensor=training_x,target_tensor=training_y)
	loader= Data.DataLoader(
		dataset=train_data_set,
		batch_size=Batch_size,
		shuffle=True);

	#put the testing data to the gpu
	te_x,te_y=var_output(testing_x,testing_y,GPU=GPU)

	if not mini_batch:

		# convert the data type to var
		tr_x,tr_y=var_output(training_x,training_y,GPU=GPU)

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
			
			#writing the error to the txt file /print it 
			file_write(text_name,net,loss_function,tr_x,tr_y,te_x,te_y,epoch,100,file_w=True)

	else:
		#The mini betch option process
		for epoch in range(iter_no):
			for betch_x,betch_y in loader:
				optimizer.zero_grad();
				tr_x,tr_y=var_output(betch_x,betch_y,GPU=GPU)
				prediction = net(tr_x),
				cost= loss_function(prediction[0],tr_y),
				cost[0].backward()
				optimizer.step()
				file_write(text_name,net,loss_function,tr_x,tr_y,te_x,te_y,epoch,100,file_w=True)
		
		tr_x=Variable(training_x).cuda()
		tr_y=Variable(training_y).float().cuda()

	old_cost=cost[0].cpu().data.numpy()
	#sometimes it will not train well, so this step try to make the cost low
	while cost_check(training_x,cost[0])>final_error:
		optimizer.zero_grad();
		prediction = net(tr_x)
		cost= loss_function(prediction,tr_y),
		cost[0].backward()
		optimizer.step()
		if cost[0].cpu().data.numpy()<old_cost:
			file_write(text_name,net,loss_function,tr_x,tr_y,te_x,te_y,1,1,file_w=True)
			old_cost=cost[0].cpu().data.numpy()


	file_write(text_name,net,loss_function,tr_x,tr_y,te_x,te_y,1,1,file_w=True)




#====================================HOW TO USE THE==========================================# 
#set the output txt name
text_name='Icy_Tin.txt'
#set the output network parameter file name
net_name='Icy_Tin.pkl'
h5_name="Icy_Tin.h5"

#get the training and testing data
training_x,training_y,testing_x,testing_y,x_max,x_min,wlen=data_to_pytorch(filepath=None,ratio=99);
net=Net(training_x.numpy().shape[1],training_y.numpy().shape[1],None)
start_time=time.time()
train_function(training_x,training_y,final_error=0.005,iter_no=1,LR=0.005,mini_batch=None,GPU=False)
net=net.cpu()


#=====================================data save===============================================#
torch.save(net.state_dict(), net_name)
fw = h5.File(h5_name, "w")
fw.create_dataset('training_x', data=training_x.numpy())
fw.create_dataset('training_y', data=training_y.numpy())
fw.create_dataset('testing_x',  data=testing_x.numpy())
fw.create_dataset('testing_y',  data=testing_y.numpy())
fw.create_dataset('x_min',      data=x_min)
fw.create_dataset('x_max',      data=x_max)
fw.create_dataset('wlen',      data=wlen)
fw.close()

file = open(text_name,'a')
file.write('time='+str(time.time()-start_time))
file.close()



