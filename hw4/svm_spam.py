
# coding: utf-8

# In[ ]:


import random
import numpy as np


# In[ ]:

def send_msg(msg="...",
             dingding_url = "https://oapi.dingtalk.com/robot/send?access_token=67f442405a74c7f0115b5e9f63da890029c7a3a41c371f436e059f1f63497eef"
             ):
    import requests
    import json
    headers = {"Content-Type": "application/json; charset=utf-8"}
    
    post_data = {
        "msgtype": "text", 
        "text": {
            "content": msg
        }
    }
    
    requests.post(dingding_url, headers=headers, 
            data=json.dumps(post_data))

from sklearn import preprocessing, metrics, linear_model
import utils
import scipy.io
import numpy as np
from linear_classifier import LinearSVM_twoclass
from sklearn.metrics import accuracy_score
        
# load the SPAM email training dataset

X,y = utils.load_mat('data/spamTrain.mat')
yy = np.ones(y.shape)
yy[y==0] = -1

# load the SPAM email test dataset

test_data = scipy.io.loadmat('data/spamTest.mat')
X_test = test_data['Xtest']
y_test = test_data['ytest'].flatten()

##################################################################################
#  YOUR CODE HERE for training the best performing SVM for the data above.       #
#  what should C be? What should num_iters be? Should X be scaled?               #
#  should X be kernelized? What should the learning rate be? What should the     #
#  number of iterations be?                                                      #
##################################################################################
best_svm = None
best_train_score = 0
best_val_score   = 0
best_test_score  = 0

svm = LinearSVM_twoclass()
svm.theta = np.zeros((X.shape[1],))
p = 0.8
X_train,X_val = X [:3200],X [3200:]
y_train,y_val = yy[:3200],yy[3200:]

# try gaussian kernel
sigma_ = 0.1
C_ = 0.3

Cvals = [0.01,0.03,0.1,0.3,1,3,10,30]
sigma_vals = [0.01,0.03,0.1,0.3,1,3,10,30]
best_C = None
best_sigma = None
bese_score = 0

K = np.load("E_distance.npy")
K_val = np.load('E_distance_val.npy')
for sigma_ in sigma_vals:
    
    K_gaussian_k = np.exp(-K/(2*(sigma_**2)))
    K_gaussian_val = np.exp(-K_val/(2*(sigma_**2)))
    # add the intercept term
    # what I want here is to iterate all the combination and get the best result for val data set.
    # 1. how to determine the best learn rate, for them to converge?
    # 2. how to save the best coefficient, so that we can reuse them later
    # 3. how to save the loss history, so that we can compate them? 
    # build a dictionary for them and save them as json.
    
    KK = np.vstack([np.ones((K_gaussian_k.shape[0],)),K_gaussian_k.T]).T
    KK_val = np.vstack([np.ones((K_gaussian_val.shape[0],)),K_gaussian_val.T]).T
        
    for C_ in Cvals:
        
        svm.theta = np.zeros((KK.shape[1],))
        
        learning_rates = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.5]
        
        best_learning_rate = 0
        best_decrease_ratio = 0
        best_svm_theta = None
        for learning_rate in learning_rates:
            svm.theta = np.zeros((KK.shape[1],))
            num_iters = 2000
            loss_history = svm.train(KK,y_train,learning_rate=learning_rate,reg=C_,num_iters=num_iters,verbose=True,batch_size=KK.shape[0])
            temp_LR = linear_model.LinearRegression()
            XXX = np.array(range(num_iters-100)).reshape(num_iters-100,1)
            yyy = np.array(loss_history)[100:]
            temp_LR.fit(X=XXX,y=yyy)
            print temp_LR.coef_
            if temp_LR.coef_[0] < best_decrease_ratio:
                best_learning_rate = learning_rate
                best_svm_theta = svm.theta
                best_decrease_ratio = temp_LR.coef_[0]
                print best_learning_rate
                print best_decrease_ratio
                
            send_msg("for sigma_ = "+(str)(sigma_) + " c_ = "+(str)(C_)+
                     ",  when learning rate is "+(str)(learning_rate)+
                     " decrease_ratio is "+(str)(temp_LR.coef_[0])+" final loss is" + (str)(loss_history[-1]))
            np.save("simga_"+(str)(sigma_*100)+"C_"+(str)(C_)+"learn"+(str)((int)(learning_rate*10000000)),np.array(loss_history))
            np.save("theta_of_simga_"+(str)(sigma_*100)+"C_"+(str)(C_)+"learn"+(str)((int)(learning_rate*10000000)),svm.theta)
            
        svm.theta = best_svm_theta
        loss_history = svm.train(KK,y_train,learning_rate=best_learning_rate,reg=C_,num_iters=12000,verbose=True,batch_size=KK.shape[0])
        loss_history = svm.train(KK,y_train,learning_rate=best_learning_rate/5.0,reg=C_,num_iters=6000,verbose=True,batch_size=KK.shape[0])
        loss_history = svm.train(KK,y_train,learning_rate=best_learning_rate/50.0,reg=C_,num_iters=6000,verbose=True,batch_size=KK.shape[0])
        
        score_ = (accuracy_score(svm.predict(KK),y_train))
        score_val = (accuracy_score(svm.predict(KK_val),y_val))
        
        np.save("bestsimga_"+(str)(sigma_*100)+"C_"+(str)(C_)+"learn"+(str)((int)(learning_rate*10000000)),np.array(loss_history))
        np.save("besttheta_of_simga_"+(str)(sigma_*100)+"C_"+(str)(C_)+"learn"+(str)((int)(learning_rate*10000000)),svm.theta)
            
        send_msg("for sigma_ = "+(str)(sigma_) + " c_ = "+(str)(C_)+
                     ",  when learning rate is "+(str)(learning_rate)+
                     " decrease_ratio is "+(str)(temp_LR.coef_[0])+
                     " final loss is " + (str)(loss_history[-1])+
                     " score of train is "+(str)(score_)+
                     " score of val is "+(str)(score_val)
            )
        if score_val > bese_score:
            bese_score = score_val
            best_C = C_
            best_sigma = sigma_
            send_msg("best score of val is "+(str)(score_val)+
                     " when sigma = "+(str)(best_sigma) +
                     " and C = " + (str)(C_))


loss_history = svm.train(KK,y_train,learning_rate=1e-4,reg=C_,num_iters=2000,verbose=True,batch_size=KK.shape[0])
temp_LR = linear_model.LinearRegression()
XXX = np.array(range(1900)).reshape(1900,1)
yyy = np.array(loss_history)[100:]
temp_LR.fit(X=XXX,y=yyy)
print(temp_LR.coef_)
##################################################################################
# YOUR CODE HERE for testing your best model's performance                       #
# what is the accuracy of your best model on the test set? On the training set?  #
##################################################################################

# calculate the largest learning rate 


##################################################################################
# ANALYSIS OF MODEL: Print the top 15 words that are predictive of spam and for  #
# ham. Hint: use the coefficient values of the learned model                     #
##################################################################################
words, inv_words = utils.get_vocab_dict()

##################################################################################
#                    END OF YOUR CODE                                            #
##################################################################################

