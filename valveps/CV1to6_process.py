def CV1to6_proc(Xnew,Y1new,Y2new,Y3new,compplotflag,errorplotflag,valvestring,filename):
    
    import pandas as pd
#    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
#    import math
#    import matplotlib.pyplot as plt
    from plot_compdiff import plot_comparison,plot_difference
#    valvestring='Valve CV1';  # for debugging
#    #current_flag0,dc_flag0,valve_flag0=CV1to6_proc(X0new,Y10new,Y20new,Y30new,compplotflag,errorplotflag,valvestring)
#    Xnew=X0new
#    Y1new=Y10new
#    Y2new=Y20new
#    Y3new=Y30new
    
    current_flagi=[0,0,0]  #initialize the error flags to be 0 first
    dc_flagi=[0,0,0]
    valve_flagi=[0,0,0]
        
    #define plot flags, if want to generate the plots, this could be for testing purpose
    compplot_flag=compplotflag;
    errorplot_flag=errorplotflag;
   
        
    #%%
    # use the differential of adjacent and imeediate two to find the events
    
    Xdiff=np.diff(Xnew);  # find the differential of the command, then will try to figure out the index  for the events
    
    Xdiff2=[];
    for i in range(len(Xnew)-2):
        Xdiff2.append(Xnew[i+2]-Xnew[i])
    Xdiff3=np.asarray(Xdiff2)  # convert the list to numpy array
    
    #%% This part detect closed valve operation and compare it with standard                  
    #closeindx1=np.asarray(np.where((Xdiff>2305) & (Xdiff<2312)))
    #closeindx1=closeindx1[0]
    #closeindx2=np.asarray(np.where((Xdiff3>2305) & (Xdiff3<2312)))
    #closeindx2=closeindx2[0]
    closeindx1=np.asarray(np.where(Xdiff==2306))
    
    closeindx2=np.asarray(np.where(Xdiff3==2306))
    
    #%%
    #if  (closeindx1.__len__()>0) and (closeindx2.__len__()>0): 
    diffcloseindx2=np.diff(closeindx2) 
    adjaindx=np.asarray(np.where(diffcloseindx2==1))  
    closeindx3=np.asarray(np.delete(closeindx2,adjaindx[1]))
    closeindx3r=np.reshape(closeindx3,(1,len(closeindx3)))
    
    list1=list(itertools.chain.from_iterable(closeindx1))
    list2=list(itertools.chain.from_iterable(closeindx3r))
    
    closeind=list(set().union(list1,list2));
                 
    #%%  check if the index event is really a close valve operation
    
    validcloseindx=[]
    
    for ind in range(len(closeind)):
        if Y3new[closeind[ind]-5]>99 and Y3new[closeind[ind]-5]<102:           
             validcloseindx.append(closeind[ind]) 
    
    #% compare the saved waveform and data and compute errors
    datacomp = pd.read_csv('cv1_close.csv')
    
    #%
    dc=datacomp["current"].values
    dd=datacomp["dc"].values
    dv=datacomp["valve"].values
    ds=datacomp["command"].values
    dc = dc.astype('float32')
    dd = dd.astype('float32')
    dv = dv.astype('float32')
    ds = ds.astype('float32')
    m=range(len(dc))
    #% plot and set the flags for close valve operation    
    operation=1  
    for indclose in range(len(validcloseindx)):         
        sectionc=Y1new[validcloseindx[indclose]:validcloseindx[indclose]+len(dc)]
        sectiond=Y2new[validcloseindx[indclose]:validcloseindx[indclose]+len(dd)]
        sectionv=Y3new[validcloseindx[indclose]:validcloseindx[indclose]+len(dv)]
        
        if compplot_flag==1:
            plot_comparison(dc,sectionc,dd,sectiond,dv,sectionv,indclose,operation,valvestring,filename)
    #     
        if errorplot_flag==1:
        # plot the difference
            plot_difference(dc,sectionc,dd,sectiond,dv,sectionv,indclose,operation,valvestring,filename)
    
        
        # error flags
        if sum(abs(dc-sectionc))/len(dc)>0.1:
            current_flagi[0]=1
        if sum(abs(dd-sectiond))/len(dd)>1.0:
            dc_flagi[0]=1
        if sum(abs(dv-sectionv))/len(dv)>2.0:
            valve_flagi[0]=1
        
        
    #%% This part detect open valve operation and compare it with standard 
        
    #openindx1=np.asarray(np.where((Xdiff>2175) & (Xdiff<2183)))
    #openindx1=openindx1[0]
    #openindx2=np.asarray(np.where((Xdiff3>2175) & (Xdiff3<2183)))
    #openindx2=openindx2[0]
    #if  (openindx1.__len__()>0) and (openindx2.__len__()>0):
        
    openindx1=np.asarray(np.where(Xdiff==2180))
    openindx2=np.asarray(np.where(Xdiff3==2180))    
    diffopenindx2=np.diff(openindx2) 
    adjaindxo=np.asarray(np.where(diffopenindx2==1))  
    openindx3=np.asarray(np.delete(openindx2,adjaindxo[1]))
    openindx3r=np.reshape(openindx3,(1,len(openindx3)))
    
    list3=list(itertools.chain.from_iterable(openindx1))
    list4=list(itertools.chain.from_iterable(openindx3r))
    
    openind=list(set().union(list3,list4));
                 
    #%  check if the index event is really an open valve operation
    
    validopenindx=[]
    
    for indo in range(len(openind)):
        if Y3new[openind[indo]-5]<5:           
             validopenindx.append(openind[indo]) 
    
    #% compare the saved waveform and data and compute errors
    del datacomp,dc,dd,dv,ds,m
    datacomp = pd.read_csv('cv1_open.csv')
    dc=datacomp["current"].values
    dd=datacomp["dc"].values
    dv=datacomp["valve"].values
    ds=datacomp["command"].values
    dc = dc.astype('float32')
    dd = dd.astype('float32')
    dv = dv.astype('float32')
    ds = ds.astype('float32')
    m=range(len(dc))    
        
        
    #% plot and set the flags for close valve operation    
    #del sectionc,sectiond,sectionv
    if 'sectionc' in locals():
        del sectionc
    if 'sectiond' in locals():
        del sectiond
    if 'sectionv' in locals():
        del sectionv
    operation=2    
    for indopen in range(len(validopenindx)):         
        sectionc=Y1new[validopenindx[indopen]:validopenindx[indopen]+len(dc)]
        sectiond=Y2new[validopenindx[indopen]:validopenindx[indopen]+len(dd)]
        sectionv=Y3new[validopenindx[indopen]:validopenindx[indopen]+len(dv)]
        
        if compplot_flag==1:
            plot_comparison(dc,sectionc,dd,sectiond,dv,sectionv,indopen,operation,valvestring,filename)
    #     
        if errorplot_flag==1:
        # plot the difference
            plot_difference(dc,sectionc,dd,sectiond,dv,sectionv,indopen,operation,valvestring,filename)
    
        
        # error flags
        if sum(abs(dc-sectionc))/len(dc)>0.1:
            current_flagi[1]=1
        if sum(abs(dd-sectiond))/len(dd)>1.0:
            dc_flagi[1]=1
        if sum(abs(dv-sectionv))/len(dv)>2.0:
            valve_flagi[1]=1    
    
    
    #%% This part detect calibration valve operation and compare it with standard 
        
    #caliind0=np.asarray(np.where((Xdiff>1000) & (Xdiff<1040)))
       
    caliind0=np.asarray(np.where((Xdiff>-3080) & (Xdiff<-3070)))
    caliind=caliind0[0] 
    #caliind=np.asarray(np.where(Xdiff==-3075))         
    #%  check if the index event is really a calibration valve operation
    
    validcaliindx=[]
    
    for indc in range(len(caliind)):
         #the previous is valve moving and calibration in progress
        if (Xnew[caliind[indc]-25]<1040) & (Xnew[caliind[indc]-25]>1000):          
             validcaliindx.append(caliind[indc]) 
    
    
    
    #% compare the saved waveform and data and compute errors
    del datacomp,dc,dd,dv,ds,m
    datacomp = pd.read_csv('cv1_calibration.csv')
    dc=datacomp["current"].values
    dd=datacomp["dc"].values
    dv=datacomp["valve"].values
    ds=datacomp["command"].values
    dc = dc.astype('float32')
    dd = dd.astype('float32')
    dv = dv.astype('float32')
    ds = ds.astype('float32')
    m=range(len(dc))    
        
        
    #% plot and set the flags for close valve operation    
    #del sectionc,sectiond,sectionv
    if 'sectionc' in locals():
        del sectionc
    if 'sectiond' in locals():
        del sectiond
    if 'sectionv' in locals():
        del sectionv
    operation=3    
    for indcali in range(len(validcaliindx)):         
    #    sectionc=Y1new[validcaliindx[indcali]:validcaliindx[indcali]+len(dc)]
    #    sectiond=Y2new[validcaliindx[indcali]:validcaliindx[indcali]+len(dd)]
    #    sectionv=Y3new[validcaliindx[indcali]:validcaliindx[indcali]+len(dv)]
        sectionc=Y1new[validcaliindx[indcali]-len(dc)+1:validcaliindx[indcali]+1]
        sectiond=Y2new[validcaliindx[indcali]-len(dc)+1:validcaliindx[indcali]+1]
        sectionv=Y3new[validcaliindx[indcali]-len(dc)+1:validcaliindx[indcali]+1]
        
        if compplot_flag==1:
            plot_comparison(dc,sectionc,dd,sectiond,dv,sectionv,indcali,operation,valvestring,filename)
    #     
        if errorplot_flag==1:     # plot the difference
            plot_difference(dc,sectionc,dd,sectiond,dv,sectionv,indcali,operation,valvestring,filename)
    
        
        # error flags
        if sum(abs(dc-sectionc))/len(dc)>0.1:
            current_flagi[2]=1
        if sum(abs(dd-sectiond))/len(dd)>1.0:
            dc_flagi[2]=1
        if sum(abs(dv-sectionv))/len(dv)>1.0:
            valve_flagi[2]=1      

 #%% final output of the flags
    
    
    return current_flagi,dc_flagi,valve_flagi