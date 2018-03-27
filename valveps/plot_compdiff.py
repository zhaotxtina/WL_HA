#import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
#import math


#%%
def plot_comparison(dc,sectionc,dd,sectiond,dv,sectionv,ind,operation,valvestring,filename):
    #% Plot the comparison of the predicted and target data
#    plt.clf()
    indx=ind+1
    plt.figure(figsize=(20,20))
    plt.subplot(311)
    l1,=plt.plot(dc,color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(sectionc,color='green', linewidth=2.0)
    plt.legend(handles=[l1, l2], labels=['predicted', 'target'],  loc='best')
    plt.title("%s %s comparison of current, duty cycle,valve position for operation %s" %(filename,valvestring,indx))
    plt.subplot(312)
    l1,=plt.plot(dd,color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(sectiond,color='green', linewidth=2.0)
    plt.legend(handles=[l1, l2], labels=['predicted', 'target'],  loc='best')
#        plt.title('Comparison of prediction to target for duty cycle')
    plt.subplot(313)
    l1,=plt.plot(dv,color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(sectionv,color='green', linewidth=2.0)
    plt.legend(handles=[l1, l2], labels=['predicted', 'target'],  loc='best')
#        plt.title('Comparison of prediction to target for valve position')
    if operation==1:
        plt.savefig("%s %s comparison for close operation %s.png"  %(filename,valvestring,indx))
    elif operation==2:
        plt.savefig("%s %s comparison for open operation %s.png"  %(filename,valvestring,indx))
    else:
        plt.savefig("%s %s comparison for calibration operation %s.png"  %(filename,valvestring,indx))
     
    return    
    
def plot_difference(dc,sectionc,dd,sectiond,dv,sectionv,ind,operation,valvestring,filename):
#    plt.clf()
    indx=ind+1
    plt.figure(figsize=(20,20))
    l1,=plt.plot(abs(dc-sectionc),color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(abs(dd-sectiond),color='green', linewidth=2.0)
    l3,=plt.plot(abs(dv-sectionv),color='cyan', linewidth=2.0)
    plt.legend(handles=[l1,l2,l3], labels=['current', 'duty cycle','valve position'],  loc='best')
    plt.title("%s %s difference from prediction to target for operation %s.png"  %(filename,valvestring,indx))
    
    if operation==1:
        plt.savefig("%s %s difference for close operation %s.png"  %(filename,valvestring,indx))
    elif operation==2:
        plt.savefig("%s %s difference for open operation %s.png"  %(filename,valvestring,indx))
    else:
        plt.savefig("%s %s difference for calibration operation %s.png"  %(filename,valvestring,indx))
    return    
   
    
