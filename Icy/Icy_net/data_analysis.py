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

'''
THIS CODE IS TELL HOW TO RELOAD THE DATA FOR ANALYSIS AFTER TRAINING 
IT HAVE THREE BASIC STEPS TO YOU IF YOU WANT TO DO THE ANALYSIS FIRST
'''

#Step1
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

#Step2.
#================================= READ THE H5 file===========================================#
path_of_h5="Icy_Flux.h5"
fr = h5.File(path_of_h5, "r")
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
PATH_of_pkl="Icy_Flux.pkl"
net=Icy_Flux(training_x.shape[1],training_y.shape[1],None)
net.load_state_dict(torch.load(PATH_of_pkl));


#=============================== HOW TO PREDICT THE VALUE======================================#
#1.set the value as the following format: x= np.array([tau,Teff,Mass,logg,CO2,Rin]).T
#2.use prediction function : predicted_spectrum=prediction(x,xmax,xmin,net)
#Then it will give give a spectrum array to you


def prediction(x):
    x=((x-xmin)*0.8/(xmax-xmin) +0.1).astype('float32');
    x=Variable(torch.unsqueeze(torch.from_numpy(x),dim=1)).contiguous().view(1,training_x.shape[1])
    net.cpu().eval()
    predict = net(x)
    return predict.data.numpy()


'''

  The following is the error checking tools which using the testing_x testing_y for you to test
the accacury of the training network.It provides three main error checking funcions

#1.  error_checking_function():The simple fast error check. It will give two array, the absulate error
difference percentage(dF/F*100) between predicted value and true value of training data set and testing 
data set. It will return the maximum error difference of each spectrum (choose the max error from 130 data
points), you can change to the 99% value one.(see the comment in that function)

#2. neig_error() : A plot of error and n dimensional negibhoor relationship

#3. plot_check(): Show the data visualization of error difference between the predicted value and true value.

'''


#=========================================== How To Use ===================================================#
#When you read the H5 file already from part 
#just use this : a,b=error_checking_function(err_term) 
#None= Max error "99%"= the 99% data point is below this error.
#a : The error array of training data
#b : The error array of testing data

def error_checking_function(err_term=None):

    def prediction_difference(x,y,net,term=None):
        x=Variable(torch.unsqueeze(torch.from_numpy(x),dim=1)).contiguous().view(1,training_x.shape[1])
        net.cpu().eval()
        predict = net(x)
        per_Error=(np.exp(y)-np.exp(predict.data.numpy()[0]))/np.exp(y)*100;
        return per_Error

    def error_choose_function(error_array,cost_array,i,err_term=None):
        if err_term==None:
                cost_array[i]=abs(error_array).max()
        elif err_term=="99%":
                cost_array[i]=np.sort(abs(error_array))[-5]
        elif err_term=="mean":
            cost_array[i]=np.mean(abs(error_array))
        elif err_term=="median":
            cost_array[i]=np.median(abs(error_array))

    cost_1,cost_2=np.zeros(len(training_x)),np.zeros(len(testing_x))
    cost_3=np.zeros((len(testing_x),len(testing_y[1])))
    for i in xrange(len(training_x)):
        max_error1=prediction_difference(training_x[i],training_y[i],net)
        error_choose_function(max_error1,cost_1,i,err_term)
    for j in xrange(len(testing_x)):
        max_error2=prediction_difference(testing_x[j],testing_y[j],net)
        error_choose_function(max_error2,cost_2,j,err_term)
        cost_3[j]=prediction_difference(testing_x[j],testing_y[j],net)

    return cost_1,cost_2



#=========================================== neig_error function =================================================#

#==========================================WHAT IS THIS? =========================================================#
#  We were training the spectrum with five/six parameters(optical depth,Teff,Rin etc), In mechine learning
# we call that features, it can always say that we were finding a spectrum function F in a R5 or R6 space
# F(Teff,Mass,Rin,logg,CO2,etc). So that we can represent the data points in a 5/6 dimensional
# space. For This function, I created a 5/6(OR EVEN) dimensional space depands on testing data set features.
# This function is trying to search the neighboors point surrounded from the testing data set. The following graph
# showed a example in a 2D space of Teff and logg. X is the data points O is the training point.
#
# ^logg * * * * * * * * * * * * * * * * * * * * * * * * * * * *   O training points
# | * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *   X data points
# | * * * * * * * * O X O * * * * * * * * * * * * * * * * * * *
# | * * * * * * * * * O * * * * * * * * * * * * * * * * * * * * 
# | * * * * * * * * * * * * * * * * * * * * * * * * * * * * * - -> Teff
#
# To search the X point , see how many training data next to it. For this example, the data point surrounded by 3 training 
# data points, I will call it have 3 negihboors .It is very important becuase the network is predicting the value from the 
# the reference of training data. The number of the training points surrounded the testing point  may determined the
# acccuray of the predicted data at that testing data. So, this function is try to plot the relation of error and number 
# of neighboors points.

###############################################WARNING############################################################
#THIS FUNCTION MAY HAVE BUGS.

#=============================================== How To Use ======================================================#
#just like: nd_error_sum,b=neig_error(testing_x,training_xtesting_y,training_y)
#It will give out the error of :
#nd_error_sum:array of how negihboor point of testing set
#           b: error differnce of testing set same as the b from 1st error_checking_function(*).
#A plot of neihboor point error

def neig_error(testing_x,training_x,testing_y,training_y):

    # convert the testing data to the nd coordinates
    def index_find(x,compare_list):
        ind_pos=np.zeros(len(compare_list[1,:]))
        for i in range(len(compare_list[1,:])):
            ind_pos[i]=np.where(compare_list[:,i]==x[i])[0]
        return ind_pos.astype(int)

    #It will give out a n dimensional array and compare list of the feature
    #The compare list will help to find the coordinates in the n dimensional space
    def nd_array(testing_x,training_x):

        #menge the training and testing set.
        referene_x=np.vstack((testing_x,training_x));

        #find the dimension of the label
        max_axis_value=np.zeros(len(referene_x[1,:]));
        unique_list=[]
        max_num=0
        for i in  xrange(len(referene_x[1,:])):
            unique=np.unique(referene_x[:,i])
            max_axis_value[i]=len(unique);
            unique_list.append(unique)
            max_num=len(unique) if len(unique)>max_num else max_num

        #creating the compare list
        compare_list=np.zeros([max_num,len(referene_x[1,:])]);
        for j in xrange(len(unique_list)):
            #mark them the value found at unquie list
            compare_list[0:len(unique_list[j]),j]=unique_list[j]

        #create the n dimenstionsal array
        ndarray=np.zeros(max_axis_value.astype(int));
        for k in xrange(len(training_x[:,1])):
            #mark down the training data into it.
            ndarray[tuple(index_find(training_x[k],compare_list))]=1
        return ndarray,compare_list

    #find the total negihboors point of that testing point 
    def nd_negihbors_sum(pos,testing_x,ndarray,compare_list):
        #find the coordinate
        pos=index_find(testing_x[pos],compare_list)
        a=[-1,1]
        value=0
        #search in each dimension is that have negihboor point
        for i in xrange(len(pos)):
            for j in xrange(len(a)):
                act=np.copy(pos)
                act[i]=act[i]+a[j] if ndarray.shape[i]>act[i]+a[j] and act[i]+a[j]>=0 else act[i]
                value=value+ndarray[tuple(act)]
        #return the total number
        return value

    a,b=error_checking_function();
    b_5=np.where(b>5)[0]
    ndarray,compare_list=nd_array(testing_x,training_x)
    nd_error_sum=np.zeros(len(b_5));
    for i, item in enumerate(b_5):
        nd_error_sum[i]=nd_negihbors_sum(item,testing_x,ndarray,compare_list)
    #testing_x2=scale_back_label(testing_x,xmin,xmax)
    #plt.plot(testing_x2[:,0][b_5],b[b_5],'o')
    plt.plot(nd_error_sum,b[b_5],'o')
    plt.xlabel('negihboors points number')
    plt.ylabel('error %')
    plt.show()
    return nd_error_sum,b

#This is a scale back function
#since the label is scale between 0.1~0.9, this is using to scale back the real value from the hdf5 file
def scale_back_label(module_x,x_min,x_max):
    back_x=(module_x-0.1)*(x_max-x_min)*1.0/0.8+x_min
    return back_x

#======================================= Checking plot function ========================================#
#   This function is a checking function to checking the value difference between predicted value and the 
#true value. It have the following plot:
#1. The loglog plot of between flux and wavelength
#2. The normal plot of between flux and wavelength   
#3. The infromation(optical depth,Teff etc) of the testing label
#4. dF/F*100% The absulate error between predicted and true value in difference wavelength

#========================================== How To Use ==================================================#
#testing_x and testing_y is the testing set which already selected from training ( store in the hdf5 file)
#index is testing_x/testing_y[index]
def plot_check(i,mode='testing'):

    if mode=='testing':
        x,y=testing_x,testing_y
    elif mode=='training':
        x,y=training_x,training_y
    else:
        Error('Not correct mode')

    def scale_back_label(module_x,x_min,x_max):
        back_x=(module_x-0.1)*(x_max-x_min)*1.0/0.8+x_min
        return back_x

    plt.subplot(2,2,1)
    v_x=Variable(torch.unsqueeze(torch.from_numpy(x[i]),dim=1)).contiguous().view(1,testing_x.shape[1])
    net.cpu().eval()
    predict = net(v_x)
    plt.loglog(wlen[1],np.exp(predict.data.numpy()[0]),'ro',label="predicted");
    plt.loglog(wlen[1],np.exp(y[i]),'bo',fillstyle='none',label="real")
    plt.xlabel('wavelength')
    plt.ylabel('Flux')
    plt.legend(loc="best")

    plt.subplot(2,2,2)
    plt.plot(wlen[1],np.exp(predict.data.numpy()[0]),'ro',label="predicted");
    plt.plot(wlen[1],np.exp(y[i]),'bo',fillstyle='none',label="real")
    plt.xlabel('wavelength')
    plt.ylabel('Flux')
    plt.legend(loc="best")

    plt.subplot(2,2,3)
    #per_Error=(np.exp(y[i])-np.exp(predict.data.numpy()[0]))/np.exp(y[i])*100;
    per_Error=(np.exp(y[i])-np.exp(predict.data.numpy()[0]))/np.exp(y[i])*100;
    plt.semilogx(wlen[1],abs(per_Error))
    plt.xlabel('wavelength')
    plt.ylabel('dF/F*100%')
    plt.legend(loc='best')
    plt.savefig('1.png')

    plt.subplot(2,2,4)
    label_back_value=scale_back_label(x[i],xmin,xmax)
    label_thing=['tau','Teff','logg','CO2','Rin']
    for index, item in enumerate(label_back_value):
        plt.text(0, 0.15*index+0.1, str(label_thing[index])+'='+str(item), fontsize=16)
    plt.legend(loc="best")
    plt.show()


################################################################## TRAINING ERROR ANALYSIS #########################################################################
#WARNING : THE FOLLOWING FUNCTION IS ONLY FOR THE 5 FEATURES MODEL IF YOU HAVE 6 OR MORE PLS ADD IT YOU ElSE


################################################################## Optical depth checking function ################################################################3
# This netowrk model is want to predict the spectrum under difference tau(Optical depth) value.
# So this function is want find the relation between the gap size of the tau value and error.
# Definition : gap size The distance between two closeest neighboor tau value under the same setting in the training data
# Like this                           1.1 <----------------->1.15<------------------->1.2           ====> Gap size= 1.2 -1.1=0.1
#                                   (traning data)      (testing data)             (training data)
#
#It will give a plot to user that the relatino between gap size and the error%
#you can choose mean mode to conbine all the data at the same gap or not.

def tau_check(err_term=None,mean_mode=None):
    def scale_back_label(module_x,x_min,x_max):
        back_x=(module_x-0.1)*(x_max-x_min)*1.0/0.8+x_min
        return back_x
    #Find the Error From the checking function 
    a,b=error_checking_function(err_term);

    #scale back the value
    lb_train_x=scale_back_label(training_x,xmin,xmax)

    lb_test_x =scale_back_label(testing_x,xmin,xmax)

    #Create a gap value array 
    gap=np.zeros(len(testing_x[:,0]))

    for i in xrange(len(testing_x[:,0])):

        # Searching under the same value(Teff,Mass,logg,etc), Is that have the two closest neighboor
        if len(np.where((lb_train_x[:,0]<lb_test_x[i,0])&(lb_train_x[:,1]==lb_test_x[i,1])&(lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,3]==lb_test_x[i,3])\
            &(lb_train_x[:,4]==lb_test_x[i,4]))[0])>0 and len(np.where((lb_train_x[:,0]>lb_test_x[i,0])&(lb_train_x[:,1]==lb_test_x[i,1])&\
            (lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,3]==lb_test_x[i,3])&(lb_train_x[:,4]==lb_test_x[i,4]))[0])>0:

            #Search the neighboor larger than checking value
            front_index=np.where((lb_train_x[:,0]>lb_test_x[i,0])&(lb_train_x[:,1]==lb_test_x[i,1])&(lb_train_x[:,2]==lb_test_x[i,2])&\
                (lb_train_x[:,3]==lb_test_x[i,3])&(lb_train_x[:,4]==lb_test_x[i,4]))[0]

            #Search the neighboor smaller than checking value
            back_index =np.where((lb_train_x[:,0]<lb_test_x[i,0])&(lb_train_x[:,1]==lb_test_x[i,1])&(lb_train_x[:,2]==lb_test_x[i,2])&\
                (lb_train_x[:,3]==lb_test_x[i,3])&(lb_train_x[:,4]==lb_test_x[i,4]))[0]

            # Find the closest value
            front_value=lb_train_x[front_index][:,0].min()
            back_value =lb_train_x[back_index ][:,0].max()

            #Cal the gap size
            gap[i]=abs(front_value-back_value);
        else:
            #If they were at the edge(just like 0,4 that dont have two cloesest neighboor),label them as 3
            gap[i]=6

    #plot them
    if mean_mode==None:
        index=np.where(b>5)[0]
        plt.plot(gap[index],b[index],'o')
        plt.ylabel('error %',fontsize=14)
    else:
        m_error=[];
        index_2=np.where(b>5)[0]
        for j in range(len(np.unique(gap))):
            index=np.where(gap==np.unique(gap)[j])[0]
            m_error.append(np.mean(b[index]))
        plt.plot(np.unique(gap),m_error,'o')
        plt.plot(np.unique(gap),m_error)
        plt.ylabel('mean error %')
    plt.xlabel('tau gap number')
    plt.ylim(0)
    if err_term==None:
        plt.title('(Max Error)Error ~ tau gap relation of testing smaple='+str(len(testing_x[:,0])))
    else:
        plt.title('(99% Error)Error ~ tau gap relation of testing smaple='+str(len(testing_x[:,0])))
    plt.show()

################################################################## Edge Error checking function ################################################################

def edge_error_check(err_term=None,mean_mode=None):

    def scale_back_label(module_x,x_min,x_max):
        back_x=(module_x-0.1)*(x_max-x_min)*1.0/0.8+x_min
        return back_x

    def fit_line(x,y):
        m,b = np.polyfit(x, y, 1)
        return m,b

    a,b=error_checking_function(err_term);
    lb_train_x=scale_back_label(training_x,xmin,xmax)
    lb_test_x =scale_back_label(testing_x,xmin,xmax)
    edge_num=np.zeros(len(testing_x[:,0]))
    for i in xrange(len(testing_x[:,0])):
        if len(np.where((lb_train_x[:,0]<lb_test_x[i,0])&(lb_train_x[:,1]==lb_test_x[i,1])&(lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,3]==lb_test_x[i,3])\
            &(lb_train_x[:,4]==lb_test_x[i,4]))[0])==0 or \
        len(np.where((lb_train_x[:,0]>lb_test_x[i,0])&(lb_train_x[:,1]==lb_test_x[i,1])&\
            (lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,3]==lb_test_x[i,3])&(lb_train_x[:,4]==lb_test_x[i,4]))[0])==0:

            edge_num[i]+=1


        if len(np.where((lb_train_x[:,1]<lb_test_x[i,1])&(lb_train_x[:,0]==lb_test_x[i,0])&(lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,3]==lb_test_x[i,3])\
            &(lb_train_x[:,4]==lb_test_x[i,4]))[0])==0 or len(np.where((lb_train_x[:,1]>lb_test_x[i,1])&(lb_train_x[:,0]==lb_test_x[i,0])&\
            (lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,3]==lb_test_x[i,3])&(lb_train_x[:,4]==lb_test_x[i,4]))[0])==0:

            edge_num[i]+=1

        if len(np.where((lb_train_x[:,2]<lb_test_x[i,2])&(lb_train_x[:,0]==lb_test_x[i,0])&(lb_train_x[:,1]==lb_test_x[i,1])&(lb_train_x[:,3]==lb_test_x[i,3])\
            &(lb_train_x[:,4]==lb_test_x[i,4]))[0])==0 or len(np.where((lb_train_x[:,2]>lb_test_x[i,2])&(lb_train_x[:,0]==lb_test_x[i,0])&\
            (lb_train_x[:,1]==lb_test_x[i,1])&(lb_train_x[:,3]==lb_test_x[i,3])&(lb_train_x[:,4]==lb_test_x[i,4]))[0])==0:

            edge_num[i]+=1

        if len(np.where((lb_train_x[:,3]<lb_test_x[i,3])&(lb_train_x[:,1]==lb_test_x[i,1])&(lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,0]==lb_test_x[i,0])\
            &(lb_train_x[:,4]==lb_test_x[i,4]))[0])==0 or len(np.where((lb_train_x[:,3]>lb_test_x[i,3])&(lb_train_x[:,1]==lb_test_x[i,1])&\
            (lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,0]==lb_test_x[i,0])&(lb_train_x[:,4]==lb_test_x[i,4]))[0])==0:

            edge_num[i]+=1

        if len(np.where((lb_train_x[:,4]<lb_test_x[i,4])&(lb_train_x[:,1]==lb_test_x[i,1])&(lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,3]==lb_test_x[i,3])\
            &(lb_train_x[:,0]==lb_test_x[i,0]))[0])==0 or len(np.where((lb_train_x[:,4]>lb_test_x[i,4])&(lb_train_x[:,1]==lb_test_x[i,1])&\
            (lb_train_x[:,2]==lb_test_x[i,2])&(lb_train_x[:,3]==lb_test_x[i,3])&(lb_train_x[:,0]==lb_test_x[i,0]))[0])==0:

            edge_num[i]+=1


    if mean_mode==None:
        plt.plot(edge_num,b,'o')
        plt.ylabel('error %')
    else:
        m_error=[];
        for j in range(len(testing_x[1,:])+1):
            index=np.where(edge_num==j)[0]
            if len(index)>0:
                m_error.append(np.mean(b[index]))
            else:
                m_error.append(0)
        m1,c1=fit_line(np.arange(len(testing_x[1,:])+1),m_error)
        plt.plot(np.arange(len(testing_x[1,:])+1),np.array(np.arange(len(testing_x[1,:])+1)*m1)+c1,label='slope='+str(m1))
        plt.plot(edge_num,b,'o',color="gray",alpha=0.3)
        plt.plot(np.arange(len(testing_x[1,:])+1),m_error,'ro',label="mean data")
        #plt.plot(np.arange(len(testing_x[1,:])+1),m_error)
        plt.ylabel('mean error %')
        plt.legend(loc="best")
    plt.xlabel('edge number')
    plt.xlim(-.5,5.5)
    plt.ylim(0)
    if err_term==None:
        plt.title('(Max Error)Error ~ edge number relation of testing smaple='+str(len(testing_x[:,0])))
    else:
        plt.title('(99% Error)Error ~ edge number relation of testing smaple='+str(len(testing_x[:,0])))
    plt.savefig('nd_edge_relation.png')
    plt.show()   



#checking the extreme value(>=5%) exist at what tau
def tau_check2(err_ter,extreme=5):
    a,b=error_checking_function(err_term);
    index=np.where(b>=extreme)[0];
    index2=np.where(a>=extreme)[0];
    lb_test_x =scale_back_label(testing_x,xmin,xmax)
    lb_train_x =scale_back_label(training_x,xmin,xmax)
    plt.plot(lb_test_x[index,0],b[index],'o')
    plt.xlim(0,4.5)
    plt.xlabel('optical depth value')
    plt.ylabel('% Error')
    plt.title('Error~optical depth relation (>5%Error)')
    plt.savefig('tau_error_5.png')
    plt.show()


def error_wlen_function():

    def prediction_difference(x,y,net,term=None):
        x=Variable(torch.unsqueeze(torch.from_numpy(x),dim=1)).contiguous().view(1,training_x.shape[1])
        net.cpu().eval()
        predict = net(x)
        per_Error=(np.exp(y)-np.exp(predict.data.numpy()[0]))/np.exp(y)*100;
        return per_Error

    def error_choose_function(error_array,cost_array,i,err_term=None):
        if err_term==None:
            cost_array[i]=np.where(abs(error_array)>5)[0]

    cost_1,cost_2=[],[]
    wlen_error_t1,wlen_error_t2=np.zeros(len(wlen[1])),np.zeros(len(wlen[1]))
    wlen_error_t3=np.zeros((len(testing_y),len(wlen[1])))
    error_3=[]
    error_4=[]
    for i in xrange(len(training_x)):
        error_array=prediction_difference(training_x[i],training_y[i],net)
        wlen_error_t1+=np.abs(error_array)

    for j in xrange(len(testing_x)):
        error_array=prediction_difference(testing_x[j],testing_y[j],net)
        wlen_error_t2+=np.abs(error_array)
        wlen_error_t3[j]=np.abs(error_array)

    for k in xrange(len(wlen[1])):
        error_3.append(wlen_error_t3[:,k].max())
        error_4.append(np.median(wlen_error_t3[:,k]))
    plt.semilogx(wlen[1],wlen_error_t2*1.0/len(testing_x),label='mean error')
    plt.semilogx(wlen[1],error_3,label='maximum error')
    plt.semilogx(wlen[1],error_4,label='medain error')
    plt.legend(loc='best')
    plt.xlabel('wavelength(um)')
    plt.ylabel(' Error %')
    plt.title('Error percentage with difference wavelength')
    plt.show()
