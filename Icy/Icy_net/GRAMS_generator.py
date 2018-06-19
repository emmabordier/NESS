#load and test/prediction module
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
from astropy.io import fits
'''
THIS CODE IS TO GENERATE THE NEW DATA FOR GRAMS
'''

################# THE GRAMS FILE/DATA SOURCE SHOULD PUT UNDER HERE##################


#===================MOVE/COPY YOU TRAINING NETWORK STRUCTURE TO HERE=========================#
class Icy_Flux(torch.nn.Module):
    def __init__(self,input_size,output_size,ner_size):
        super(Icy_Flux, self).__init__()
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

class Icy_Tin(torch.nn.Module):
    def __init__(self,input_size,output_size,ner_size):
        super(Icy_Tin, self).__init__()
        self.input_layer         =torch.nn.Linear(input_size,4*input_size, bias=True)
        self.hidden_feature_1    =torch.nn.Linear(4*input_size,output_size, bias=True)
        self.predict_layer       =torch.nn.Linear(output_size,output_size, bias=True)

    def forward(self,x):
        x=F.sigmoid(self.input_layer(x))
        x=F.sigmoid(self.hidden_feature_1(x))
        x=self.predict_layer(x)
        return x


#Step2.
#================================= READ THE H5 file===========================================#
path_of_h5file="Icy_Flux.h5"
fr = h5.File(path_of_h5file, "r")
training_x=fr['training_x'].value
training_y=fr['training_y'].value
testing_x =fr['testing_x'].value
testing_y =fr['testing_y'].value
xmax      =fr['x_max'].value
xmin      =fr['x_min'].value
wlen      =fr['wlen'].value
fr.close()


#Step 3.
#=============================== move back the parameter=======================================#
PATH='Icy_Flux.pkl'
PATH2='Icy_Tin.pkl'
Flux_net=Icy_Flux(training_x.shape[1],training_y.shape[1],None)
Flux_net.load_state_dict(torch.load(PATH));
Tin_net=Icy_Tin(training_x.shape[1],1,None)
Tin_net.load_state_dict(torch.load(PATH2));

##################################################################################################

def New_GRAMS(x_max,x_min,filepath=None,filename="Extend_GRAMS_C.fits"):
    text_name="rejected.txt"
    def prediction(x,x_max,x_min,net):
        x=((x-x_min)*0.8/(x_max-x_min) +0.1);
        x=Variable(torch.unsqueeze(torch.from_numpy(x),dim=1).float()).contiguous().view(1,training_x.shape[1])
        net.cpu().eval()
        predict = net(x)
        return np.exp(predict.data.numpy())

    def ss_readfits(fitsfile):
        if fitsfile[-5:] != '.fits': 
            fitsfile+='.fits'
        data=Table.read(fitsfile)
        tags=data.colnames
        return data, tags


    #filter the function from the grams data(NOT USED AT THIS CODE)
    def Filter_function(T,g,COR,R,Ms,data):
        return np.where((data['C2O']==COR)&(data['Teff']==T)&(data['Rin']==R)&(data['Mass']==Ms)&(data['logg']==g))[0]
    


    if filepath is None:
        filepath='grams_c.fits';
    
    #reading the data from fits file
    data,tags=ss_readfits(filepath);
    #=============================================old data===============================================================#
    old_Teff    =data['Teff']
    old_logg    =data['logg']
    old_Mass    =data['Mass']
    old_C2O     =data['C2O']
    old_Rin     =data['Rin']
    old_tau11_3 =data['tau11_3']
    old_tau1    =data['tau1']
    old_Lum     =data['Lum']
    old_MLR     =data['MLR']
    old_Fphot   =data['Fphot']
    old_mphot   =data['mphot']
    old_Tin     =data['Tin']
    old_Lspec   =data['Lspec'] 
    old_Fspec   =data['Fspec']
    old_Fstar   =data['Fstar']

    #array of optical depths for which to generate new grid
    index_pos=np.where(old_Mass>=2.0)[0]
    before_1=np.linspace(0.15,0.95,9);
    after_1=np.linspace(1.1,3.4,24);
    gen_data=[];
    for T in np.unique(old_Teff[index_pos]):
        for g in np.unique(old_logg[index_pos]):
            for COR in np.unique(old_C2O[index_pos]):
                for R in np.unique(old_Rin[index_pos]):
                    index=Filter_function(T,g,COR,R,2.0,data)
                    if len(index)>0:
                        if old_tau11_3[index].max()>1.0:
                            for tau_value in before_1:
                                if tau_value<old_tau11_3[index].max():
                                    gen_data.append(np.array([tau_value,T,g,COR,R]))
                            for tau_value in after_1:
                                if tau_value%0.5!=0 and tau_value<old_tau11_3[index].max():
                                    gen_data.append(np.array([tau_value,T,g,COR,R]))
                        elif old_tau11_3[index].max()<1.0:
                            for tau_value in before_1:
                                if tau_value<old_tau11_3[index].max():
                                    gen_data.append(np.array([tau_value,T,g,COR,R]))



    gen_data=np.array(gen_data);

    new_Teff    =[]
    new_logg    =[]
    new_Mass    =[]
    new_C2O     =[]
    new_Rin     =[]
    new_tau11_3 =[]    
    new_tau1    =[]# Y
    new_Lum     =[]
    new_MLR     =[]# Y
    new_Fphot   =[]# Y
    new_mphot   =[]# Y
    new_Tin     =[]# Y
    new_Lspec   =[]# Y
    new_Fspec   =[]# Y
    new_Fstar   =[]# Y
    for x in gen_data:
        index=Filter_function(x[1],x[2],x[3],x[4],2,data)
        #xb=np.array((i,2800,-0.2,2,7)).T
        check_Tin=prediction(x,xmax,xmin,Tin_net)
        if (check_Tin)<1850:
            new_Teff.append(x[1])
            new_logg.append(x[2])
            new_Mass.append(2.0)
            new_C2O.append(x[3])
            new_Rin.append(x[4])
            new_tau11_3.append(x[0])
            new_Tin.append(check_Tin[0][0])
            new_Lum.append(old_Lum[index[0]])
            new_Fspec.append(prediction(x,xmax,xmin,Flux_net)[0])
            new_MLR.append(x[-1]/0.01*old_MLR[index][np.where(old_tau11_3[index]==0.01)[0]][0])
            new_tau1.append(x[-1]/0.01*old_tau1[index][np.where(old_tau11_3[index]==0.01)[0]][0])
            new_Fphot.append(np.full_like(old_Fphot[1], np.nan, dtype=np.double))
            new_mphot.append(np.full_like(old_Fphot[1], np.nan, dtype=np.double))
            new_Lspec.append(wlen[1])
            new_Fstar.append(old_Fstar[index[1]])
        else:
            print(x)
            file = open(text_name,'a')
            file.write('Reject: tau='+str(x[0])+' ,Teff='+str(x[1])+' ,log(g)='+str(x[2])+' , CO2='+str(x[3])+' ,Rin='+str(x[4]))
            file.close()

    New_GRAMS_file = Table([np.hstack((old_Teff,np.array(new_Teff))),\
            np.hstack((old_logg,np.array(new_logg))),\
            np.hstack((old_Mass,np.array(new_Mass))),\
            np.hstack((old_C2O,np.array(new_C2O))),\
            np.hstack((old_Rin,np.array(new_Rin))),\
            np.hstack((old_tau11_3,np.array(new_tau11_3))),\
            np.hstack((old_tau1,np.array(new_tau1))),\
            np.hstack((old_Lum,np.array(new_Lum))),\
            np.hstack((old_MLR,np.array(new_MLR))),\
            np.vstack((old_Fphot,np.array(new_Fphot))),\
            np.vstack((old_mphot,np.array(new_mphot))),\
            np.hstack((old_Tin,np.array(new_Tin))),\
            np.vstack((old_Lspec,np.array(new_Lspec))),\
            np.vstack((old_Fspec,np.array(new_Fspec))),\
            np.vstack((old_Fstar,np.array(new_Fstar)))],names=('Teff','logg','Mass','C2O','Rin','tau11_3','tau1','Lum','MLR','Fphot','mphot','Tin','Lspec','Fspec','Fstar'))
    New_GRAMS_file.write(filename, format='fits')

    print('End of produce')
New_GRAMS(xmax,xmin,filepath=None,filename="Extend_GRAMS_C.fits")
