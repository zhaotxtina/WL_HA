import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # for reproducibilit
import math
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,RMSprop
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#%%
#Sta006_df5 = Sta006_df5.apply(lambda x: preprocessing.scale(x))
#Sta006_df5 = pd.read_csv('ctf1run1usvlas4trainJan16.csv')
Sta006_df5 = pd.read_csv('ctf1run1sta006sm.csv')
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

#%% did some manipulating on the data to make it cleaner and logic correct
X[50:100]=0.0;
Y[50:100,:]=0.0;
Y[50:100,1]=1.0;
plt.plot(X)
plt.plot(Y)
plt.show()

Xm1=X[175:400]
Xm2=X[1:50]
Xm3=np.concatenate((Xm1, Xm2), axis=0)
plt.plot(Xm3)
Ym1=Y[175:400,:]
Ym2=Y[1:50,:]
Ym3=np.concatenate((Ym1, Ym2), axis=0)
plt.plot(Ym3)
Xm=Xm3
Ym=Ym3
#%%
#% another set of data for testing
#input1 = pd.read_csv('simplefitInputs.csv')
#output1 = pd.read_csv('simplefitTargets.csv')
#X=input1.values
#Y=output1.values
#X = X.astype('float32')
#Y = Y.astype('float32')
#%
#X = np.random.rand(100)
#Y = 5 * X + np.random.rand(100)
#%
scaler = MinMaxScaler(feature_range=(0, 1))
Xs = scaler.fit_transform(Xm)
Ys = scaler.fit_transform(Ym)
#X2s=np.column_stack((X1s, Ys))
#%
#plt.plot(Xs)
#plt.plot(Ys)
#plt.plot(Xs[280:340])
#plt.plot(Ys[280:340])
#plt.show()
#ax.set_title('input to output')
#X1s = numpy.reshape(X1s, (X1s.shape[0], 1))
#X1s = numpy.reshape(X1s, (X1s.shape[0], 1, X1s.shape[1]))
#% create and fit the  network
#X_train,Y_train=Xs[:1250],Ys[:1250]
#X_test,Y_test=Xs[1250:],Ys[1250:]
X_train,Y_train=Xs,Ys
#X_train,Y_train=Xs[280:340],Ys[280:340]
#%%
#X_train.reshape(-1, 1) 
#X_test,Y_test=Xs[60:],Ys[60:]
# fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
##%% worked for current and duty cycle  and simple net exampel data
#model=Sequential()
#model.add(Dense(output_dim=20,input_dim=1,activation='relu'))
#model.add(Dense(output_dim=20,activation='relu'))
#model.add(Dense(output_dim=20,activation='relu'))
#model.add(Dense(output_dim=20,activation='relu'))
#model.add(Dense(output_dim=20,activation='relu'))
#model.add(Dense(output_dim=20,activation='relu'))
#model.add(Dense(output_dim=20,activation='relu'))
#model.add(Dense(output_dim=20,activation='relu'))
#model.add(Dense(output_dim=20,activation='relu'))
#model.add(Dense(output_dim=20,activation='relu'))
##model.add(Dense(output_dim=10,input_dim=1,activation='tanh'))
##model.add(Dense(output_dim=10,activation='tanh'))
##model.add(Dense(output_dim=10,activation='tanh'))
##model.add(Dense(output_dim=10,activation='tanh'))
##model.add(Dense(output_dim=10,activation='tanh'))
##model.add(Dense(output_dim=10,activation='tanh'))
##model.add(Dense(output_dim=8,activation='tanh'))
##model.add(Dense(output_dim=5,activation='tanh'))
#model.add(Dense(1))
#model.add(Activation('tanh'))
##model.add(Dense(1,activation='softmax'))
##model.compile(loss='mse',optimizer='sgd')
#sgd = SGD(lr=0.06, decay=1e-8, momentum=0.5, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd)
##model.fit(X_train,Y_train, nb_epoch=500, batch_size=50)
##Y_pred=model.predict(X_train)

#%%  This is for valve position fitting
model=Sequential()
model.add(Dense(output_dim=40,input_dim=1,activation='tanh'))
model.add(Dense(output_dim=30,activation='tanh'))
model.add(Dense(output_dim=30,activation='tanh'))
model.add(Dense(output_dim=30,activation='tanh'))
model.add(Dense(output_dim=20,activation='tanh'))
model.add(Dense(output_dim=20,activation='tanh'))
model.add(Dense(output_dim=10,activation='tanh'))
model.add(Dense(output_dim=10,activation='tanh'))
model.add(Dense(output_dim=10,activation='tanh'))
model.add(Dense(3))
#model.add(Activation('relu'))
#model.add(Dense(1,activation='softmax'))
#model.compile(loss='mse',optimizer='sgd')
#sgd = SGD(lr=0.07, decay=1e-8, momentum=0.5, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd)
#rmsprop=RMSprop(lr=0.01, rho=0.9, epsilon=1e-04, decay=1e-6)
rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #default value
model.compile(loss='mean_squared_error', optimizer='rmsprop')

#adagrad=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
#model.compile(loss='mean_squared_error', optimizer='adagrad')

#model.fit(X_train,Y_train, nb_epoch=500, batch_size=50)
##%
#Y_pred=model.predict(X_train)
##%
#predY = scaler.inverse_transform(Y_pred)
#trainY = scaler.inverse_transform(Y_train)
#%
#import matplotlib
#matplotlib.style.use('ggplot')
#fig, axes = plt.subplots(nrows=1, ncols=2)
#predY[:,0].plot(ax=axes[0,0]); axes[0,0].set_title('predicted position');
#trainY[:,0].plot(ax=axes[0,1]); axes[0,1].set_title('target position');
##Sta006_df4['USVPOSMS1'].plot(ax=axes[1,0]); axes[1,0].set_title('USVPOSMS1');
#import pylab
#plt.plot(predY[:,2])
#plt.plot(trainY[:,2])
#%
#plt.plot(predY)
#plt.plot(trainY)
#Y_pred=model.predict(X_train)
##%
#predY = scaler.inverse_transform(Y_pred)
#trainY = scaler.inverse_transform(Y_train)
#plt.plot(Y_pred)
#plt.plot(Y_train)
##plt.scatter(X_train,Y_train)
##plt.scatter(X_train,Y_pred)
#plt.show()
#pylab.savefig('dutyc_predJan23.pdf')
#%
#model.fit(X_train,Y_train,batch_size=40)

#training
print('Training.....')
for step in range(8001):
    cost= model.train_on_batch(X_train,Y_train)
    if step % 100 == 0:
        print('train cost:',cost)

Y_pred=model.predict(X_train)
#%
predY = scaler.inverse_transform(Y_pred)
trainY = scaler.inverse_transform(Y_train)
#% evaluate the model

scores = model.evaluate(X_train,Y_train)
print("%s: %.2f%%" % (model.metrics_names, scores*100))

#scores = model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#%
plt.plot(predY[:,0])
plt.plot(trainY[:,0])
##plt.scatter(X_train,Y_train)
##plt.scatter(X_train,Y_pred)
plt.show()
#Y_pred=model.predict(X_train)
#plt.scatter(X_train,Y_train)
#plt.scatter(X_train,Y_pred)
#plt.show()        
##%       
##test
#print('\n Testing ......')
#cost= model.evaluate(X_test,Y_test,batch_size=40)
#%% save the tained NN model
from keras.models import model_from_json
import os
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#%% load the NN model
from keras.models import model_from_json
import os
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
#%%
#modle=loaded_model
loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')
#%%
Y_pred=loaded_model.predict(X_train)
#%
predY = scaler.inverse_transform(Y_pred)
trainY = scaler.inverse_transform(Y_train)
#%%
plt.plot(predY[:,0])
plt.plot(trainY[:,0])
##plt.scatter(X_train,Y_train)
##plt.scatter(X_train,Y_pred)
plt.show()
#%%

Ediff=predY-trainY
l1,=plt.plot(Ediff[:,0],color='red', linewidth=2.0, linestyle='--')
l2,=plt.plot(Ediff[:,1],color='green', linewidth=2.0)
l3,=plt.plot(Ediff[:,2],color='blue', linewidth=1.0, linestyle='-.')
plt.legend(handles=[l1, l2,l3], labels=['current', 'duty cycle','valve position'],  loc='best')
plt.title('error from prediction to target')
plt.show()
#%%
#output the smoothed difference
def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
#%%
#Ehat = sgolay2d( Ediff, window_size=11, order=3)
Ehat0 = savitzky_golay(Ediff[:,0], 15,0)
Ehat1 = savitzky_golay(Ediff[:,1], 15,0)
Ehat2 = savitzky_golay(Ediff[:,2], 15,0)


#%%
l1,=plt.plot(Ehat0,color='red', linewidth=2.0, linestyle='--')
l2,=plt.plot(Ehat1,color='green', linewidth=2.0)
l3,=plt.plot(Ehat2,color='blue', linewidth=1.0, linestyle='-.')
plt.legend(handles=[l1, l2,l3], labels=['current', 'duty cycle','valve position'],  loc='best')
plt.title('Smoothed error from prediction to target')
plt.show()
#%% output a status for 


#%%
scores = loaded_model.evaluate(X_train,Y_train)
print("%s: %.2f%%" % (loaded_model.metrics_names, scores*100))
#%%

##%%
#Y_pred2=model.predict(X_test)
##Predict = scaler.inverse_transform(Y_pred)
##Xtest=scaler.inverse_transform(X_test)
##Ytest=scaler.inverse_transform(Y_test)
##trainY = scaler.inverse_transform(Ys)
##trainX = scaler.inverse_transform(X1s)
##plt.scatter(Xtest,Y_pred)
#plt.scatter(X_test,Y_test)
#plt.scatter(X_test,Y_pred2)
#plt.show()
##Y_pred=model.predict(X_test)
##plt.plot(Y_pred)
##plt.plot(Y_test)
##plt.show()
##%%
#Y_pred=model.predict(X_train)
#plt.scatter(X_train,Y_train)
#plt.scatter(X_train,Y_pred)
#plt.show()

##%%
#
#model = Sequential()
#
#model = Sequential([
#    Dense(32, input_dim=1),
#    Activation('relu'),
#    Dense(10),
#    Activation('relu'),
#    Dense(10),
#    Activation('relu'),
#    Dense(10),
#    Activation('relu'),
#    Dense(10),
#    Activation('relu'),
#    Dense(1),
#    Activation('softmax'),
#])
##model.add(Dense(20,activation='relu'))
##model.add(Dense(20,activation='relu'))
##model.add(Dense(20,activation='relu'))
##model.add(Dense(10,activation='relu'))
##model.add(Dense(10,activation='relu'))
##model.add(Dense(10,activation='relu'))
##model.add(Dense(10,activation='relu'))
##model.add(Dense(1))     
##sgd=SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)     
##model.compile(loss='mse', optimizer='sgd',metrics=['accuracy'])
##
#model.compile(loss='mse',optimizer='adam')
#model.fit(Xs, Ys, nb_epoch=20, batch_size=16)   #, verbose=2)
#score = model.evaluate(Xs, Ys, batch_size=16)
##%% batch training
#print('Training.....')
#for step in range(101):
#    cost= model.train_on_batch(Xs,Ys)
#    if step % 100 == 0:
#        print('train cost:',cost)
#
##%%
##test
##print('\n Testing ......')
##cost= model.evaluate(Xs,Ys,batch_size=40)
##print('test cost:',cost)
##W,b=model.layers[0].get_weights()
##print('Weights=',W,'\nbiases=',b)
#
##%%plot the prediction
#Y_pred=model.predict(Xs)
#plt.scatter(Xs,Ys)
##plt.plot(X1s,Ys)
##plt.plot(Y_pred)
#plt.show()
##%%
#plt.plot(Xs)
##plt.plot(X)
##plt.plot(Ys)
#plt.plot(Y_pred)
##plt.plot(trainX)
#plt.show()
##%%
##Yp = model.predict(X1s)
##plt.plot(Yp)
##plt.plot(Ys)
##%%
##Predict = scaler.inverse_transform(Y_pred)
##trainY = scaler.inverse_transform(Ys)
##trainX = scaler.inverse_transform(X1s)
#
#plt.plot(Xs)
##plt.plot(X)
#plt.plot(Ys)
##plt.plot(trainX)
#plt.show()
##%%
#plt.scatter(Xs,Ys)
##%%
#plt.scatter(Xs,Y_pred)


#