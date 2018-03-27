import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
#import math

import matplotlib.pyplot as plt
import time
#import os, time, datetime, pickle, json
from datetime import datetime
import math



import csv
import re
import string

from os import listdir
from os.path import isfile, join

#%pylab inline
#%matplotlib inline


#%% Populating the interactive namespace from numpy and matplotlib

out_dir = './Result/'
sections = ['VER', 'WELL', 'PAR', 'CURVE', 'VAL']
#pat_ver_start = re.compile(r'~Version') # Version section start pattern
#pat_well_start = re.compile(r'~Well') # Well section start pattern
#pat_par_start = re.compile(r'~Parameter') # Parameter section start pattern
pat_curve_start = re.compile(r'~Log_Definition') # Curve section start pattern
pat_cval_start = re.compile(r'~Log_Data')
#pat_cval_start = re.compile(r'~Log_Data[1] \s') # Curve value section start pattern
pat_end = re.compile(r'#-{40,}') # Section end pattern
#pat_delim1 = re.compile('\.') # first delimimator
#pat_delim2 = re.compile('\s\:') # The second deliminator
#pat_delim3 = re.compile('\s{2,}') # Delimimator between col2 and col3
pat_cval =  re.compile(r'[^\s]+') # Channel values pattern
offset_ver = 1
offset_well = 3
offset_par = 3
offset_curve = 3
offset_cval = 1

well_list = ['WR540', 'WR584']


#%%
#start_ver = pat_idx(df, 'LINE', pat_ver_start)[0] + offset_ver
            
def pat_idx(df, col_name, pat):
    df['PAT_FLG'] = df[col_name].apply(lambda x: pd.notnull(pat.match(x)))
    return df[df.PAT_FLG == True].index.tolist()



#%%
#path=dirName = 'C:\\Users\\tzhao\\Documents\\FNPO\\datafromNoor_Jan\\June_newformat_lasfile\\Native LAS samples\\'

def read_LAS_data(path, fn,  pat_curve_start = pat_curve_start, pat_cval_start = pat_cval_start, \
                  pat_end = pat_end, offset_ver = offset_ver, offset_well = offset_well, offset_par = offset_par, \
                  offset_curve = offset_curve, offset_cval = offset_cval, pat_cval = pat_cval):
    file_name = path + fn
   
    
    df = pd.read_csv(file_name, delimiter='\n', header = None)
    df.columns = ['LINE']

    start_curve = pat_idx(df, 'LINE', pat_curve_start)
    start_cval = pat_idx(df, 'LINE', pat_cval_start)
    
    for i in range(len(start_curve)-1):
        start_cval_copy=start_cval[i]
        end_cval_copy=start_curve[i+1]
        df_cval = df.copy()[start_cval_copy+1:end_cval_copy][['LINE']]  
          
    
        df_cval = pd.DataFrame.from_records(list(df_cval.LINE.apply(lambda x: tuple(re.findall(pat_cval, x)))))
        df_cval.columns = df_cval.iloc[0] 
        df_cval=df_cval[1:]
#        df_cval.columns = list(df_curve.ATTR)
            
#        fn1= fn.replace(".las", ".csv")
        filename= path + fn[:-4] +"_"+str(i)+".csv"
        df_cval.to_csv(filename)
        
    i= len(start_curve)-1    
    start_cval_copy=start_cval[i]
    df_cval = df.copy()[start_cval_copy+1:][['LINE']] 
    
    df_cval = pd.DataFrame.from_records(list(df_cval.LINE.apply(lambda x: tuple(re.findall(pat_cval, x)))))
    df_cval.columns = df_cval.iloc[0]
    df_cval=df_cval[1:]
#        df_cval.columns = list(df_curve.ATTR)
            
#        fn1= fn.replace(".las", ".csv")
    filename= path + fn[:-4] +"_"+str(i)+".csv"
    df_cval.to_csv(filename)


    #%%  USe tkinter to select input file
    
from Tkinter import Tk
from tkFileDialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename) 
 
path=os.path.dirname(filename)+"/" 
fn= os.path.basename(filename)
#fn=listoflas[i]
#    
read_LAS_data(path, fn,  pat_curve_start = pat_curve_start, pat_cval_start = pat_cval_start, \
              pat_end = pat_end, offset_ver = offset_ver, offset_well = offset_well, offset_par = offset_par, \
              offset_curve = offset_curve, offset_cval = offset_cval, pat_cval = pat_cval)
   
    #%%
    
##path=dirName = 'C:\\Users\\tzhao\\Documents\\FNPO\\datafromNoor_Jan\\June_newformat_lasfile\\Native LAS samples\\'
##path=dirName = 'G:\\FNPO\\labintegrationtest\\PHM_WL_2017.05.26-10.02.03_2017.05.26-10.07.29\\'
##path=dirName =  'G:\\FNPO\\labintegrationtest\\unprocessed\\PHM_WL_ftng_FCRF_FISO_FNFM_FNPC_FSCM_2017.06.01-11.18.59_2017.06.01-12.36.03\\'
#path=dirName = 'C:\\Users\\tzhao\\Documents\\FNPO\\newformatpythoncodes\\testdata\\'
##path=dirName = 'G:\\FTNG\\NewLASformatread\\'
#listOfCommands = os.listdir(dirName)
#listoflas = [t for t in listOfCommands if '.LAS' in t]
#
##os.path.join(dirName,command+'.txt')
#
#for i in range(len(listoflas)):
#    
#    fn=listoflas[i]
#    
#    read_LAS_data(path, fn,  pat_curve_start = pat_curve_start, pat_cval_start = pat_cval_start, \
#                  pat_end = pat_end, offset_ver = offset_ver, offset_well = offset_well, offset_par = offset_par, \
#                  offset_curve = offset_curve, offset_cval = offset_cval, pat_cval = pat_cval)



#%%  read the input data for detection

#listOfcsvfiles = os.listdir(path)
# select the FSCM files first
listOfcsvfiles = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and \
         'FSCM' in i]
# select the csv with _0 and _1 ending, which are for two different sampling frequencies
fn1 = [t for t in listOfcsvfiles if '_0.csv' in t]
fn2 = [t for t in listOfcsvfiles if '_1.csv' in t]

file_name1 = path + fn1[0]
file_name2 = path + fn2[0]
datain0 = pd.read_csv(file_name1)
datain1 = pd.read_csv(file_name2)
datain1=datain1[:-1]
#%% convert excel time format to standard time

import datetime


def xldate_to_datetime(xldate):
    tempDate = datetime.datetime(1900, 1, 1)
    deltaDays = datetime.timedelta(days=int(xldate))
    secs = (int((xldate%1)*86400)-60)
    detlaSeconds = datetime.timedelta(seconds=secs)
    TheTime = (tempDate + deltaDays + detlaSeconds )
    return TheTime.strftime("%Y-%m-%d %H:%M:%S")
#datetime.datetime.strptime(date1, '%Y:%m:%d %H:%M:%S')
import xlrd
 
datain0['datetime'] = [xldate_to_datetime(a) for a in datain0['#TIME_1900'].values]
datain1['datetime'] = [xldate_to_datetime(a) for a in datain1['#TIME_1900'].values]

#datain0['DateFormatted'] = [datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S') for a in datain0['datetime'].values]     



#%%  read in the needed channels to process
    # this data missing CV4  'CV4STATUSSCM',
X0=datain1[['CV1HALLVLTSCM','CV1HE1VLTSCM','CV1HE2VLTSCM','CV1HE3VLTSCM','CV1TEMPSCM','datetime']] #CV1
X1=datain1[['CV2HALLVLTSCM','CV2HE1VLTSCM','CV2HE2VLTSCM','CV2HE3VLTSCM','CV2TEMPSCM','datetime']] #CV2
X2=datain1[['CV3HALLVLTSCM','CV3HE1VLTSCM','CV3HE2VLTSCM','CV3HE3VLTSCM','CV3TEMPSCM','datetime']] #CV3
X3=datain1[['CV4HALLVLTSCM','CV4HE1VLTSCM','CV4HE2VLTSCM','CV4HE3VLTSCM','CV4TEMPSCM','datetime']] #CV4
X4=datain1[['CV5HALLVLTSCM','CV5HE1VLTSCM','CV5HE2VLTSCM','CV5HE3VLTSCM','CV5TEMPSCM','datetime']] #CV5
X5=datain1[['CV6HALLVLTSCM','CV6HE1VLTSCM','CV6HE2VLTSCM','CV6HE3VLTSCM','CV6TEMPSCM','datetime']] #CV6
X6=datain1[['USVHALLVLTSCM','USVHE1VLTSCM','USVHE2VLTSCM','USVHE3VLTSCM','USVTEMPSCM','datetime']] #USV
X7=datain1[['LSVHALLVLTSCM','LSVHE1VLTSCM','LSVHE2VLTSCM','LSVHE3VLTSCM','LSVTEMPSCM','datetime']] #LSV
X8=datain1[['MCTEMPSCM','PSHVTEMPSCM','PSRCTEMPSCM','PSTEMPSCM','PSTRNTEMPSCM']]  #temperature channels
X9=datain0[['CV1STATUSSCM','CV2STATUSSCM','CV3STATUSSCM','CV5STATUSSCM','CV6STATUSSCM','USVSTATUSSCM','LSVSTATUSSCM','datetime']]

#%%  plot function

def plot_comparison(data,valve,title):
    #% Plot the comparison of the predicted and target data
#    plt.clf()
#    data=X0
#    valve='CV1'
    plt.figure(figsize=(20,20))
#    plt.subplot(311)
    l1,=plt.plot(data[data.columns[0]],color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(data[data.columns[1]],color='green', linewidth=2.0)
    l3,=plt.plot(data[data.columns[2]],color='blue', linewidth=2.0)
    l4,=plt.plot(data[data.columns[3]],color='yellow', linewidth=2.0, linestyle='--')
#    l5,=plt.plot(data[data.columns[4]],color='cyan', linewidth=2.0)
#    l6,=plt.plot(data[data.columns[5]],color='black', linewidth=2.0)
    
    plt.legend(handles=[l1,l2,l3,l4], labels=['Hall voltage', 'HEV1','HEV2','HEV3'],  loc='best')
    plt.title("%s channels %s" %(valve,title))

#        plt.savefig("%s comparison for calibration operation %s.png"  %(valvestring,indx))
     
    return       
         
#%%  plot the channels before the despike
plot_comparison(X0,'CV1','before despike')
plot_comparison(X1,'CV2','before despike')
plot_comparison(X2,'CV3','before despike')
plot_comparison(X3,'CV4','before despike')
plot_comparison(X4,'CV5','before despike')
plot_comparison(X5,'CV6','before despike')
plot_comparison(X6,'USV','before despike')
plot_comparison(X7,'LSV','before despike')

# plot the temperature channels
plt.figure(figsize=(20,20))
l1,=plt.plot(X8[X8.columns[0]],color='red', linewidth=2.0, linestyle='--')
l2,=plt.plot(X8[X8.columns[1]],color='green', linewidth=2.0)
l3,=plt.plot(X8[X8.columns[2]],color='blue', linewidth=2.0)
l4,=plt.plot(X8[X8.columns[3]],color='yellow', linewidth=2.0, linestyle='--')
l5,=plt.plot(X8[X8.columns[4]],color='cyan', linewidth=2.0)
plt.legend(handles=[l1,l2,l3,l4,l5], labels=['MCTEMPSCM','PSHVTEMPSCM','PSRCTEMPSCM','PSTEMPSCM','PSTRNTEMPSCM'],  loc='best')
plt.title("temperature channels")
#%% remove the spikes from original measurement
def remove_spike(data1024,data64,channelid):
#    data=X0
#    id=0   # id=0 fpr CV1-CV6, 1 for USV or LSV, not sure it's true now
#   channel id=0 for cv1, 1 for cv2 etc...
    data=data1024
    stat=data64
    
    data=data.dropna()  # drop nan value
  
# remove rows with data =-999.25 for 1024ms data
    data=data.loc[data.iloc[:,0]!=-999.25]
    data=data.loc[data.iloc[:,1]!=-999.25]
    data=data.loc[data.iloc[:,2]!=-999.25]
    data=data.loc[data.iloc[:,3]!=-999.25]
    data=data.loc[data.iloc[:,4]!=-999.25]

   # now find the time for status word activation and remove from hall voltages etc
   # remove the invalid status word 65535
#   stat=X9  
    stat=stat.loc[stat.iloc[:,0]<=20000]  
    stat=stat.loc[stat.iloc[:,1]<=20000]  
    stat=stat.loc[stat.iloc[:,2]<=20000]  
    stat=stat.loc[stat.iloc[:,3]<=20000]  
    stat=stat.loc[stat.iloc[:,4]<=20000]  
    stat=stat.loc[stat.iloc[:,5]<=20000]  
    stat=stat.loc[stat.iloc[:,6]<=20000]  
   
   # keep the rows for 64 ms data with the status word on
    stat=stat.loc[stat.iloc[:,channelid]>3.0]  
    
   # take the time from these data
    excludetime=list(stat.datetime.unique())
    
    # should I exclude those if the neighboring points are all in the exclude list
    
    extratime=[]
    for ind in range(len(excludetime)-1):
        start_dt = datetime.datetime.strptime(excludetime[ind], "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.datetime.strptime(excludetime[ind+1], "%Y-%m-%d %H:%M:%S")
        delta= (end_dt-start_dt).total_seconds()
        if ((delta>1) & (delta < 10)):
            for dex in range(int(delta)-1):
                str1='0:0:%s' % str(dex+1)
                dt1=datetime.datetime.strptime(excludetime[ind], "%Y-%m-%d %H:%M:%S")+pd.to_timedelta(str1)
                time1='{:%Y-%m-%d %H:%M:%S}'.format(dt1)
                extratime.append(time1)
                
#               datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%
    excludetotal=excludetime+extratime
    excludetotal.sort()

   # exclude the time in the 1024ms data
    for ind in range(len(excludetotal)):
        data=data.loc[data.iloc[:,5]!=excludetotal[ind]]

   
#             

#    
    data.reset_index(inplace=True,drop=True)          

    return data

#%%  call the despike and plot the channels again

CV1data=remove_spike(X0,X9,0)
CV2data=remove_spike(X1,X9,1)
CV3data=remove_spike(X2,X9,2)
CV4data=remove_spike(X3,X9,3)
CV5data=remove_spike(X4,X9,3)
CV6data=remove_spike(X5,X9,4)
USVdata=remove_spike(X6,X9,5)
LSVdata=remove_spike(X7,X9,6)

# plot them again
plot_comparison(CV1data,'CV1','after despike')
plot_comparison(CV2data,'CV2','after despike')
plot_comparison(CV3data,'CV3','after despike')
plot_comparison(CV4data,'CV4','after despike')
plot_comparison(CV5data,'CV5','after despike')
plot_comparison(CV6data,'CV6','after despike')
plot_comparison(USVdata,'USV','after despike')
plot_comparison(LSVdata,'LSV','after despike')

#%% apply the Hall effect voltage check
def hallvolt_check(data,valve):
#    data=CV2data   # for debugging
    hall_flag=[0,0,0];
    forhall_comb=[0   for b in data[data.columns[0]]]
      
    # apply voltage level logic first
    # hall effect voltage has to be 
    hall_flag0=[0  if b>=10.4  else  1   for b in data[data.columns[0]] ]
    data[data.columns[0]]=[1  if b>=10.4  else  0   for b in data[data.columns[0]] ]
    hall_flag[0]=1 if sum(hall_flag0)>=1 else 0
   
     # check HEV1-3 should be between 2 to 9.4 V
    for j in range(4)[1:4]:
        for i in range(len(data)):
            if data.iat[i,j]>=9.4:
                data.iat[i,j]=1.0
            elif data.iat[i,j]<=2.0:
                data.iat[i,j]=0.0
            else:
                data.iat[i,j]=0.5
                    

    
              
    # now check those marked with 0.5 to see if an error is triggered
    for i in range(len(data))[1:(len(data)-1)]:    
        for j in range(4)[1:4]: 
            if  data.iat[i,j]==0.5:
                if ( (data.iat[i-1,j]==0.0 and data.iat[i+1,j]==1.0) or (data.iat[i-1,j]==1.0 and data.iat[i+1,j]==0.0) or \
                 (data.iat[i-1,j]==0.0 and data.iat[i+1,j]==0.0) or (data.iat[i-1,j]==1.0 and data.iat[i+1,j]==1.0) or \
                 (data.iat[i-1,j]==0.0 and data.iat[i+1,j]==0.5   and data.iat[i+2,j]==1.0 ) or \
                 (data.iat[i-2,j]==0.0 and data.iat[i-1,j]==0.5   and data.iat[i+1,j]==1.0 ) or \
                 (data.iat[i-1,j]==1.0 and data.iat[i+1,j]==0.5   and data.iat[i+2,j]==0.0 ) or \
                 (data.iat[i-2,j]==1.0 and data.iat[i-1,j]==0.5   and data.iat[i+1,j]==0.0 ) or \
                 (data.iat[i-1,j]==0.0 and data.iat[i+1,j]==0.5   and data.iat[i+2,j]==0.0 ) or \
                 (data.iat[i-2,j]==0.0 and data.iat[i-1,j]==0.5   and data.iat[i+1,j]==0.0 ) or \
                 (data.iat[i-1,j]==1.0 and data.iat[i+1,j]==0.5   and data.iat[i+2,j]==1.0 ) or \
                 (data.iat[i-2,j]==1.0 and data.iat[i-1,j]==0.5   and data.iat[i+1,j]==1.0 ) #or \
#                 (data.iat[i-1,j]==0.0 and data.iat[i+1,j]==0.5 and data.iat[i+2,j]==0.5  and data.iat[i+3,j]==1.0 ) or \
#                 (data.iat[i-2,j]==0.0 and data.iat[i-1,j]==0.5 and data.iat[i+1,j]==0.5  and data.iat[i+2,j]==1.0 ) or \
#                 (data.iat[i-3,j]==0.0 and data.iat[i-2,j]==0.5 and data.iat[i-1,j]==0.5  and data.iat[i+1,j]==1.0 ) or \
#                 (data.iat[i-1,j]==1.0 and data.iat[i+1,j]==0.5 and data.iat[i+2,j]==0.5  and data.iat[i+3,j]==0.0 ) or \
#                 (data.iat[i-2,j]==1.0 and data.iat[i-1,j]==0.5 and data.iat[i+1,j]==0.5  and data.iat[i+2,j]==0.0 ) or \
#                 (data.iat[i-3,j]==1.0 and data.iat[i-2,j]==0.5 and data.iat[i-1,j]==0.5  and data.iat[i+1,j]==0.0 ) or \
#                 (data.iat[i-1,j]==0.0 and data.iat[i+1,j]==0.5 and data.iat[i+2,j]==0.5  and data.iat[i+3,j]==0.0 ) or \
#                 (data.iat[i-2,j]==0.0 and data.iat[i-1,j]==0.5 and data.iat[i+1,j]==0.5  and data.iat[i+2,j]==0.0 ) or \
#                 (data.iat[i-3,j]==0.0 and data.iat[i-2,j]==0.5 and data.iat[i-1,j]==0.5  and data.iat[i+1,j]==0.0 ) or \
#                 (data.iat[i-1,j]==1.0 and data.iat[i+1,j]==0.5 and data.iat[i+2,j]==0.5  and data.iat[i+3,j]==1.0 ) or \
#                 (data.iat[i-2,j]==1.0 and data.iat[i-1,j]==0.5 and data.iat[i+1,j]==0.5  and data.iat[i+2,j]==1.0 ) or \
#                 (data.iat[i-3,j]==1.0 and data.iat[i-2,j]==0.5 and data.iat[i-1,j]==0.5  and data.iat[i+1,j]==1.0 ) 
                ):
                    data.iat[i,j]=0.5
                else:
                    data.iat[i,j]=0.5
                    hall_flag[1]=1
        if ( (data.iat[i,1]==1 and data.iat[i,2]==1 and data.iat[i,3]==1) or \
             (data.iat[i,1]==0 and data.iat[i,2]==0 and data.iat[i,3]==0) ):
            forhall_comb[i]=1
        else:
            forhall_comb[i]=0
    # now check combination logic
    #for hall_comb flag check, if 1, indicate violation, but if it's only 1-2 duration, it's ok
    del forhall_comb[0] 
    del forhall_comb[-1]  
    indxhall=np.where(np.diff(forhall_comb[1:])>0)
    indxhallcomb=list(itertools.chain.from_iterable(indxhall))
    if len(indxhallcomb)>0:
        for p in range(len(indxhallcomb)):
            if indxhallcomb[p]<=2:
                del forhall_comb[indxhallcomb[p]:indxhallcomb[p]+2]
            else:
                del forhall_comb[indxhallcomb[p]-2:indxhallcomb[p]+2]
  
    hall_flag3=[1 if b==1.0 else 0.0 for b in forhall_comb]
    
    hall_flag[2]= 1 if sum(hall_flag3)>=1 else 0 
     
    plt.figure(figsize=(20,20))
#    plt.subplot(311)
    l1,=plt.plot(data[data.columns[0]],color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(data[data.columns[1]],color='green', linewidth=2.0)
    l3,=plt.plot(data[data.columns[2]],color='blue', linewidth=2.0)
    l4,=plt.plot(data[data.columns[3]],color='yellow', linewidth=2.0, linestyle='--')
    
    
    plt.legend(handles=[l1,l2,l3,l4], labels=['Hall voltage', 'HEV1','HEV2','HEV3'],  loc='best')
    plt.title("%s HEV channels " %(valve))
    plt.savefig("%s HEV channels.png"  %(valve))         
    return hall_flag      
#%%
#current_flag,dc_flag,valve_flag=USVLSV_proc(Xnew,Y1new,Y2new,Y3new,compplotflag,errorplotflag)
valvestring='Valve CV1';
hall_flag0=hallvolt_check(CV1data,valvestring)
#current_flag0,dc_flag0,valve_flag0=hallvolt_check(CV1data,compplotflag,errorplotflag,valvestring)
valvestring='Valve CV2';
hall_flag1=hallvolt_check(CV2data,valvestring)
valvestring='Valve CV3';
hall_flag2=hallvolt_check(CV3data,valvestring)
valvestring='Valve CV4';
hall_flag3=hallvolt_check(CV4data,valvestring)
valvestring='Valve CV5';
hall_flag4=hallvolt_check(CV5data,valvestring)
valvestring='Valve CV6';
hall_flag5=hallvolt_check(CV6data,valvestring)
valvestring='Valve USV';
hall_flag6=hallvolt_check(USVdata,valvestring)
valvestring='Valve LSV';
hall_flag7=hallvolt_check(LSVdata,valvestring)

#%% 
def output_hallflag(hall_flag,fd):
    
    if (hall_flag[0]==1):
        fd.write('Hall effect voltage: Fail\n\n')
        fd.write('\n')
    else:
        fd.write('Hall effect voltage: Pass\n\n')
        fd.write('\n')
    
    if (hall_flag[1]==1):
        fd.write('HE1to3 voltage: Fail\n\n')
        fd.write('\n')
    else:
        fd.write('HE1to3  voltage: Pass\n\n')
        fd.write('\n')
        
    if (hall_flag[2]==1):
        fd.write('HE1to3 combination voltage: Fail\n\n')
        fd.write('\n')
    else:
        fd.write('HE1to3 combination voltage: Pass\n\n')
        fd.write('\n')
    
    
    return
                    

 
#%%


#%%  final output of the flags

fileid = open(path+'\halleffectflagsoutput.txt','w');
             
fileid.write('CV1 Valve:\n\n')
output_hallflag(hall_flag0,fileid)

fileid.write('CV2 Valve:\n\n')
output_hallflag(hall_flag1,fileid)

fileid.write('CV3 Valve:\n\n')
output_hallflag(hall_flag2,fileid)

fileid.write('CV4 Valve:\n\n')
output_hallflag(hall_flag3,fileid)

fileid.write('CV5 Valve:\n\n')
output_hallflag(hall_flag4,fileid)

fileid.write('CV6 Valve:\n\n')
output_hallflag(hall_flag5,fileid)

fileid.write('USV Valve:\n\n')
output_hallflag(hall_flag6,fileid)

fileid.write('LSV Valve:\n\n')
output_hallflag(hall_flag7,fileid)


fileid.close()


#%%  if you want to clear all plots

#plt.close("all")


