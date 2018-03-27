import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # for reproducibilit
import math
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#%
#Sta006_df5 = Sta006_df5.apply(lambda x: preprocessing.scale(x))
Sta006_df5 = pd.read_csv('ctf1run1usvlas4trainJan16.csv')
#Sta006_df5 = pd.read_csv('ctf1run1sta006sm.csv')
#fig, axes = plt.subplots(nrows=2, ncols=2)
#Sta006_df5['USVMTRCURMS1'].plot(ax=axes[0,0]); axes[0,0].set_title('USVMTRCURMS1');
#Sta006_df5['USVMTRDCMS1'].plot(ax=axes[0,1]); axes[0,1].set_title('USVMTRDCMS1');
#Sta006_df5['USVPOSMS1'].plot(ax=axes[1,0]); axes[1,0].set_title('USVPOSMS1');
#Sta006_df5['USVSTATUSMS1'].plot(ax=axes[1,1]); axes[1,1].set_title('USVSTATUSMS1');
## save plot as a file
#fig.savefig('usvtraining.pdf')
#%


Y=Sta006_df5[[ "USVMTRCURMS1","USVMTRDCMS1","USVPOSMS1"]].values
#Y=Sta006_df5["USVPOSMS1"].values
X=Sta006_df5["USVSTATUSMS1"].values
#Y=Sta006_df5["USVMTRDCMS1"].values
#Y=Sta006_df5["USVPOSMS1"].values
X = X.astype('float32')
Y = Y.astype('float32')

#% did some manipulating on the data to make it cleaner and logic correct
X[210:240]=0.0;
Y[210:240,0]=0.0;# remove all the spikes and 
Y[210:240,1]=1.0;  
plt.plot(X)
plt.plot(Y)
plt.show()

#%
scaler = MinMaxScaler(feature_range=(0, 1))
Xs = scaler.fit_transform(X)
Ys = scaler.fit_transform(Y)

#% create and fit the  network
#X_train,Y_train=Xs[:1200],Ys[:1200]
#X_test,Y_test=Xs[1200:],Ys[1200:]
X_train,Y_train=Xs[230:1200],Ys[230:1200]  # only the closing/open operation
X_test,Y_test=Xs[1200:],Ys[1200:]
#X_train,Y_train=Xs,Ys
plt.plot(X_train)
#plt.plot(Ys)
#plt.show()

#%%  This is for trainging the NN
model=Sequential()
model.add(Dense(output_dim=40,input_dim=1,activation='relu'))
model.add(Dense(output_dim=30,activation='relu'))
model.add(Dense(output_dim=30,activation='relu'))
model.add(Dense(output_dim=30,activation='relu'))
model.add(Dense(output_dim=30,activation='relu'))
model.add(Dense(output_dim=20,activation='relu'))
model.add(Dense(output_dim=20,activation='relu'))
model.add(Dense(output_dim=20,activation='relu'))
model.add(Dense(output_dim=10,activation='relu'))
model.add(Dense(output_dim=10,activation='relu'))
model.add(Dense(output_dim=10,activation='relu'))
model.add(Dense(3))
#model.add(Activation('relu'))
#model.add(Dense(1,activation='softmax'))
#model.compile(loss='mse',optimizer='sgd')
#sgd = SGD(lr=0.07, decay=1e-8, momentum=0.5, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd)
#rmsprop=RMSprop(lr=0.01, rho=0.9, epsilon=1e-04, decay=1e-6)
optifun=["rmsprop"]
#lrlist=[10.0, 1.0, 0.1,0.01,0.001,0.0001]
#lrlist=[0.1, 0.01]
#rholist=[0.7]
#for x in range(len(lrlist)):

for x in range(len(optifun)):
    plt.clf()
#    rmsprop=RMSprop(lr=lrlist[x], rho=0.9, epsilon=1e-08, decay=0.0) #default value
#    rmsprop=RMSprop(lr=lrlist[x],epsilon=1e-08, decay=0.0) #default value
#    model.compile(loss='mean_squared_error', optimizer=optifun[x])
    rmsprop=RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

#adagrad=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
#model.compile(loss='mean_squared_error', optimizer='adagrad')

#model.fit(X_train,Y_train, nb_epoch=500, batch_size=50)
##%
#Y_pred=model.predict(X_train)


#training
    print('Training.....')
    for step in range(30001):
        cost= model.train_on_batch(X_train,Y_train)
        if step % 300 == 0:
            print('train cost:',cost)
        
#%
    Y_pred=model.predict(X_train)
#%
    predY = scaler.inverse_transform(Y_pred)
    trainY = scaler.inverse_transform(Y_train)
#% evaluate the model

    scores = model.evaluate(X_train,Y_train)
    print("%s: %.2f%%" % (model.metrics_names, scores*100))


#% Plot the comparison of the predicted and target data
    plt.figure(figsize=(20,20))
    plt.subplot(311)
    l1,=plt.plot(predY[:,0],color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(trainY[:,0],color='green', linewidth=2.0)
#plt.legend(handles=[l1, l2], labels=['predicted', 'target'],  loc='best')
#plt.title('for current for trainging')
    plt.title("reluoptimizer=%s " %optifun[x])   # figure out how to show variables in plots
#    plt.title(" adagrad lr=%s " %lrlist[x])   # figure out how to show variables in plots
    plt.subplot(312)
    l1,=plt.plot(predY[:,1],color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(trainY[:,1],color='green', linewidth=2.0)
#plt.legend(handles=[l1, l2], labels=['predicted', 'target'],  loc='best')
#plt.title('for duty cycle for training')
    plt.subplot(313)
    l1,=plt.plot(predY[:,2],color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(trainY[:,2],color='green', linewidth=2.0)
#plt.legend(handles=[l1, l2], labels=['predicted', 'target'],  loc='best')
#plt.title('for valve position for training')
#plt.text(2,0.5,'optimizer=%s' %optifun)
    #plt.show()

#    flag='reluoptimfun%s' % optifun[x]
    flag='opencloseonly' 
    plt.savefig("pred%s.png" %flag)    # need to differentiate 
#    flag='adagradlr%s' % lrlist[x]
#    plt.savefig("pred%s.png" %flag)    # need to differentiate 

#% predict on testing data
    Y_pred2=model.predict(X_test)

    predY2 = scaler.inverse_transform(Y_pred2)
    testY = scaler.inverse_transform(Y_test)
#% evaluate the model

    scores = model.evaluate(X_test,Y_test)
    print("%s: %.2f%%" % (model.metrics_names, scores*100))

    plt.clf()
    plt.figure(figsize=(20,20))
    plt.subplot(311)
    l1,=plt.plot(predY2[:,0],color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(testY[:,0],color='green', linewidth=2.0)
    plt.title("reluoptimizer=%s " %optifun[x])   # figure out how to show variables in plots
#    plt.title("adagrad lr=%s " %lrlist[x])   # figure out how to show variables in plots
    plt.subplot(312)
    l1,=plt.plot(predY2[:,1],color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(testY[:,1],color='green', linewidth=2.0)

    plt.subplot(313)
    l1,=plt.plot(predY2[:,2],color='red', linewidth=2.0, linestyle='--')
    l2,=plt.plot(testY[:,2],color='green', linewidth=2.0)

    plt.savefig("test%s.png" %flag)    # need to differentiate 
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% save the tained NN model
from keras.models import model_from_json
import os
# serialize model to JSON
modelopenclose_json = model.to_json()
with open("modelopenclose.json", "w") as json_file:
    json_file.write(modelopenclose_json)
# serialize weights to HDF5
model.save_weights("modelopenclose.h5")
print("Saved model to disk")
#
#%% load the NN model
from keras.models import model_from_json
import os
json_file = open('modelcali.json', 'r')
cali_model_json = json_file.read()
json_file.close()
cali_model = model_from_json(loaded_model_json)
# load weights into new model
cali_model.load_weights("modelcali.h5")
print("Loaded calibration model from disk")
#%%
#modle=loaded_model
loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')
##%%
#Y_pred=loaded_model.predict(X_train)
##%
#predY = scaler.inverse_transform(Y_pred)
#trainY = scaler.inverse_transform(Y_train)
##%%
#plt.plot(predY[:,0])
#plt.plot(trainY[:,0])
###plt.scatter(X_train,Y_train)
###plt.scatter(X_train,Y_pred)
#plt.show()
##%%
#plt.plot(predY[:,2])
#plt.plot(trainY[:,2])
#plt.show()
##%%
#plt.plot(predY[:,1])
#plt.plot(trainY[:,1])
#plt.show()
#
##%%
#
#Ediff=predY-trainY
#l1,=plt.plot(Ediff[:,0],color='red', linewidth=2.0, linestyle='--')
#l2,=plt.plot(Ediff[:,1],color='green', linewidth=2.0)
#l3,=plt.plot(Ediff[:,2],color='blue', linewidth=1.0, linestyle='-.')
#plt.legend(handles=[l1, l2,l3], labels=['current', 'duty cycle','valve position'],  loc='best')
#plt.title('error from prediction to target')
#plt.show()
##%%
##output the smoothed difference
#def savitzky_golay(y, window_size, order, deriv=0, rate=1):
#
#    import numpy as np
#    from math import factorial
#
#    try:
#        window_size = np.abs(np.int(window_size))
#        order = np.abs(np.int(order))
#    except ValueError, msg:
#        raise ValueError("window_size and order have to be of type int")
#    if window_size % 2 != 1 or window_size < 1:
#        raise TypeError("window_size size must be a positive odd number")
#    if window_size < order + 2:
#        raise TypeError("window_size is too small for the polynomials order")
#    order_range = range(order+1)
#    half_window = (window_size -1) // 2
#    # precompute coefficients
#    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
#    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
#    # pad the signal at the extremes with
#    # values taken from the signal itself
#    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
#    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
#    y = np.concatenate((firstvals, y, lastvals))
#    return np.convolve( m[::-1], y, mode='valid')
##%%
##Ehat = sgolay2d( Ediff, window_size=11, order=3)
#Ehat0 = savitzky_golay(Ediff[:,0], 15,0)
#Ehat1 = savitzky_golay(Ediff[:,1], 15,0)
#Ehat2 = savitzky_golay(Ediff[:,2], 15,0)
#
#
##%%
#l1,=plt.plot(Ehat0,color='red', linewidth=2.0, linestyle='--')
#l2,=plt.plot(Ehat1,color='green', linewidth=2.0)
#l3,=plt.plot(Ehat2,color='blue', linewidth=1.0, linestyle='-.')
#plt.legend(handles=[l1, l2,l3], labels=['current', 'duty cycle','valve position'],  loc='best')
#plt.title('Smoothed error from prediction to target')
#plt.show()
##%% output a status for 
#
#
##%%
#scores = loaded_model.evaluate(X_train,Y_train)
#print("%s: %.2f%%" % (loaded_model.metrics_names, scores*100))
##%%
