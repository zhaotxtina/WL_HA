import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import math
from USVLSV_process import USVLSV_proc
from CV1to6_process import CV1to6_proc
import matplotlib.pyplot as plt

#%%  read the input data for detection

#datain = pd.read_csv('FSCM003sta010.csv')
path = 'C:\\Users\\tzhao\\Documents\\FNPO\\labintegration\\templatesignal\\'
fn1='PHM_WL_ftng_FSCM_003_2017.07.31-02.35.34_2017.07.31-03.40.21_0.csv'

#fn3='PHM_WL_ftng_FSCM_003_2017.07.31-09.40.04_2017.07.31-09.46.26_0.csv'
fn2='PHM_WL_ftng_FSCM_ENP_2017.07.12-18.13.02_2017.07.12-18.34.16_0.csv'
file_name1 = path + fn1
datain = pd.read_csv(file_name1)
#%%  read in the needed channels to process
#         
X0=datain["CV1STATUSSCM"].values
Y10=datain["CV1MTRCURSCM"].values  #need to check if the dataset has CV1MTRCURSCM1 as the name
Y20=datain["CV1MTRDCSCM"].values
Y30R=datain["CV1RPOSSCM"].values
Y30= (Y30R-768.0)*100/(13325.0-768.0)
#
X1=datain["CV2STATUSSCM"].values
Y11=datain["CV2MTRCURSCM"].values  #need to check if the dataset has CV1MTRCURSCM1 as the name
Y21=datain["CV2MTRDCSCM"].values
Y31R=datain["CV2RPOSSCM"].values
Y31= (Y31R-768.0)*100/(13325.0-768.0)
#          
X2=datain["CV3STATUSSCM"].values
Y12=datain["CV3MTRCURSCM"].values  #need to check if the dataset has CV1MTRCURSCM1 as the name
Y22=datain["CV3MTRDCSCM"].values
Y32R=datain["CV3RPOSSCM"].values
Y32= (Y32R-768.0)*100/(13325.0-768.0)
#
X3=datain["CV4STATUSSCM"].values
Y13=datain["CV4MTRCURSCM"].values  #need to check if the dataset has CV1MTRCURSCM1 as the name
Y23=datain["CV4MTRDCSCM"].values
Y33R=datain["CV4RPOSSCM"].values
Y33= (Y33R-768.0)*100/(13325.0-768.0)
          
X4=datain["CV5STATUSSCM"].values
Y14=datain["CV5MTRCURSCM"].values  #need to check if the dataset has CV1MTRCURSCM1 as the name
Y24=datain["CV5MTRDCSCM"].values
Y34R=datain["CV5RPOSSCM"].values
Y34=(Y34R-768.0)*100/(13325.0-768.0)

X5=datain["CV6STATUSSCM"].values
Y15=datain["CV6MTRCURSCM"].values  #need to check if the dataset has CV1MTRCURSCM1 as the name
Y25=datain["CV6MTRDCSCM"].values
Y35R=datain["CV6RPOSSCM"].values
Y35=(Y35R-768.0)*100/(13325.0-768.0)

X6=datain["USVSTATUSSCM"].values
Y16=datain["USVMTRCURSCM"].values   #need to check if the dataset has CV1MTRCURSCM1 as the name
Y26=datain["USVMTRDCSCM"].values
Y36R=datain["USVRPOSSCM"].values
Y36=(Y36R-768.0)*100/(10560.0-768.0)

X7=datain["LSVSTATUSSCM"].values
Y17=datain["LSVMTRCURSCM"].values   #need to check if the dataset has CV1MTRCURSCM1 as the name
Y27=datain["LSVMTRDCSCM"].values
Y37R=datain["LSVRPOSSCM"].values
Y37=(Y37R-768.0)*100/(10560.0-768.0)
#         
#%% remove the spikes from original measurement
def remove_spike(X1,Y1,Y2,Y3):
#    X1=X6
#    Y1=Y16
#    Y2=Y26
#    Y3=Y36
    X1ind=np.where(X1>10000.0)   # find the status word value > 10000, will exclude
    Y1ind=np.where(Y1==-999.25)  # find those NA values index
    Y2ind=np.where(Y2==-999.25)
    Y3ind=np.where(Y3==-999.25)
    X1nan=np.argwhere(np.isnan(X1))
    Y1nan=np.argwhere(np.isnan(Y1))
    Y2nan=np.argwhere(np.isnan(Y2))
    Y3nan=np.argwhere(np.isnan(Y3))
    #%
    import itertools
    X1indlist=list(itertools.chain.from_iterable(X1ind))
    Y1indlist=list(itertools.chain.from_iterable(Y1ind))
    Y2indlist=list(itertools.chain.from_iterable(Y2ind))
    Y3indlist=list(itertools.chain.from_iterable(Y3ind))
    
    remove1ind=list(set().union(X1indlist,Y1indlist,Y2indlist,Y3indlist));  # merge all the spike index
    removelnan=np.unique(np.concatenate((X1nan, Y1nan,Y2nan,Y3nan), axis=0))
    removeall=np.unique(np.concatenate((remove1ind,removelnan), axis=0))
               
    X1new=np.delete(X1,removeall);
    Y1new=np.delete(Y1,removeall);
    Y2new=np.delete(Y2,removeall);    
    Y3new=np.delete(Y3,removeall);
    return X1new,Y1new,Y2new,Y3new
#
#%% 
def print_status(current_flag,dc_flag,valve_flag):
    if (1 in current_flag):
        flagcur='failure'
    else:
        flagcur='Pass'
    print "detect current error: %s" % flagcur
    if (1 in dc_flag):
        flagdc='failure'
    else:
        flagdc='pass'
    print "detect duty cycle error: %s" % flagdc
    if (1 in valve_flag):
        flagval='failure'
    else:
        flagval='Pass'
    print "detect valve position error: %s" % flagval
    
    return
#                    
#%%  apply to all the channels for eight groups
X0new,Y10new,Y20new,Y30new=remove_spike(X0,Y10,Y20,Y30)
X1new,Y11new,Y21new,Y31new=remove_spike(X1,Y11,Y21,Y31)
X2new,Y12new,Y22new,Y32new=remove_spike(X2,Y12,Y22,Y32)
X3new,Y13new,Y23new,Y33new=remove_spike(X3,Y13,Y23,Y33)
X4new,Y14new,Y24new,Y34new=remove_spike(X4,Y14,Y24,Y34)
X5new,Y15new,Y25new,Y35new=remove_spike(X5,Y15,Y25,Y35)
X6new,Y16new,Y26new,Y36new=remove_spike(X6,Y16,Y26,Y36)
X7new,Y17new,Y27new,Y37new=remove_spike(X7,Y17,Y27,Y37)
#  
#
##%%  plot the 4 channels-current,duty cycle,valve posistion and status word
#plt.figure(figsize=(20,20))
#l1,=plt.plot(Y1,color='red', linewidth=2.0, linestyle='--')
#l2,=plt.plot(Y2,color='green', linewidth=2.0)
#l3,=plt.plot(Y3,color='blue', linewidth=1.0, linestyle='-.')
#l4,=plt.plot(X,color='cyan', linewidth=1.0, linestyle='--')
#plt.legend(handles=[l1, l2,l3,l4], labels=['current', 'duty cycle','valve position','status word'],  loc='best')
#plt.title('channel plots before despiking')
#plt.show()
#
#%%
##%  for checking the signals, don't know if I should include or not
def plot_channels(Y1,Y2,Y3,X,filename,valvestring):
    plt.figure(figsize=(20,20))
    l1,=plt.plot(Y1,color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(Y2,color='green', linewidth=2.0)
    l3,=plt.plot(Y3,color='blue', linewidth=1.0, linestyle='-.')
    l4,=plt.plot(X,color='cyan', linewidth=1.0, linestyle='--')
    plt.legend(handles=[l1, l2,l3,l4], labels=['current', 'duty cycle','valve position','status word'],  loc='best')
    plt.title("%s %s channel plots after despiking"  %(filename,valvestring))
    plt.savefig("%s %s channel plots after despiking.png"  %(filename,valvestring))
    plt.show()
              
#plot the channels
filename=fn2
valvestring='Valve CV1';
plot_channels(Y10new,Y20new,Y30new,X0new,filename,valvestring)
valvestring='Valve CV2';
plot_channels(Y11new,Y21new,Y31new,X1new,filename,valvestring)
valvestring='Valve CV3';
plot_channels(Y12new,Y22new,Y32new,X2new,filename,valvestring)
valvestring='Valve CV4';
plot_channels(Y13new,Y23new,Y33new,X3new,filename,valvestring)
valvestring='Valve CV5';
plot_channels(Y14new,Y24new,Y34new,X4new,filename,valvestring)
valvestring='Valve CV6';
plot_channels(Y15new,Y25new,Y35new,X5new,filename,valvestring)
valvestring='Valve USV';
plot_channels(Y16new,Y26new,Y36new,X6new,filename,valvestring)
valvestring='Valve LSV';
plot_channels(Y17new,Y27new,Y37new,X7new,filename,valvestring)


#
#%%
#current_flag=[0,0,0]  #initialize the error flags to be 0 first
#dc_flag=[0,0,0]
#valve_flag=[0,0,0]

#define plot flags, if want to generate the plots, this could be for testing purpose
compplotflag=1;  # 0 no plot, 1 plot  comparison of the predict and target channels
errorplotflag=1;  # difference of predict and target channels

# 
##%%
#current_flag,dc_flag,valve_flag=USVLSV_proc(Xnew,Y1new,Y2new,Y3new,compplotflag,errorplotflag)
filename=fn2
valvestring='Valve CV1';
current_flag0,dc_flag0,valve_flag0=CV1to6_proc(X0new,Y10new,Y20new,Y30new,compplotflag,errorplotflag,valvestring,filename)
valvestring='Valve CV2';
current_flag1,dc_flag1,valve_flag1=CV1to6_proc(X1new,Y11new,Y21new,Y31new,compplotflag,errorplotflag,valvestring,filename)
valvestring='Valve CV3';
current_flag2,dc_flag2,valve_flag2=CV1to6_proc(X2new,Y12new,Y22new,Y32new,compplotflag,errorplotflag,valvestring,filename)
valvestring='Valve CV4';
current_flag3,dc_flag3,valve_flag3=CV1to6_proc(X3new,Y13new,Y23new,Y33new,compplotflag,errorplotflag,valvestring,filename)
valvestring='Valve CV5';
current_flag4,dc_flag4,valve_flag4=CV1to6_proc(X4new,Y14new,Y24new,Y34new,compplotflag,errorplotflag,valvestring,filename)
valvestring='Valve CV6';
current_flag5,dc_flag5,valve_flag5=CV1to6_proc(X5new,Y15new,Y25new,Y35new,compplotflag,errorplotflag,valvestring,filename)
valvestring='Valve USV';
current_flag6,dc_flag6,valve_flag6=USVLSV_proc(X6new,Y16new,Y26new,Y36new,compplotflag,errorplotflag,valvestring,filename)
valvestring='Valve LSV';
current_flag7,dc_flag7,valve_flag7=USVLSV_proc(X7new,Y17new,Y27new,Y37new,compplotflag,errorplotflag,valvestring,filename)

#
#%%  final output of the flags
#
print "for CV1 valve status:"
print_status(current_flag0,dc_flag0,valve_flag0)

print "for CV2 valve status:"
print_status(current_flag1,dc_flag1,valve_flag1)

print "for CV3 valve status:"
print_status(current_flag2,dc_flag2,valve_flag2)

print "for CV4 valve status:"
print_status(current_flag3,dc_flag3,valve_flag3)

print "for CV5 valve status:"
print_status(current_flag4,dc_flag4,valve_flag4)

print "for CV6 valve status:"
print_status(current_flag5,dc_flag5,valve_flag5)

print "for USV valve status:"
print_status(current_flag6,dc_flag6,valve_flag6)

print "for LSV valve status:"
print_status(current_flag7,dc_flag7,valve_flag7)
#
#
