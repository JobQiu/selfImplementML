
# coding: utf-8

# In[ ]:


import random
import numpy as np


# In[ ]:

def send_msg(msg="...",
             dingding_url = "https://oapi.dingtalk.com/robot/send?access_token=67f442405a74c7f0115b5e9f63da890029c7a3a41c371f436e059f1f63497eef"
             ):
    
    """
    this method is used to notify me about the results after the training
    """
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
from sklearn.model_selection import train_test_split
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
y_test_ = np.ones(y_test.shape)
y_test_[y_test == 0] = -1
y_test = y_test_

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
X_train, X_val, y_train, y_val = train_test_split(X,yy,test_size = 0.2)

# try gaussian kernel

Cvals = [0.3,1,3,10,30,50,70,80,90]
sigma_vals = [44444]
best_C = None
best_sigma = None
bese_score = 0
best_svm = None

for sigma_ in sigma_vals:

    K_gaussian_k = X_train
    K_gaussian_val = X_val

    KK = np.vstack([np.ones((K_gaussian_k.shape[0],)),K_gaussian_k.T]).T
    KK_val = np.vstack([np.ones((K_gaussian_val.shape[0],)),K_gaussian_val.T]).T
    KK_test = np.vstack([np.ones((X_test.shape[0],)),X_test.T]).T
    for C_ in Cvals:

        svm = LinearSVM_twoclass()
        svm.theta = np.zeros((KK.shape[1],))

        best_learning_rate = 1e-3
        best_decrease_ratio = 0
        loss_history = svm.train(KK,y_train,learning_rate=best_learning_rate,reg=C_,num_iters=12000,verbose=True,batch_size=KK.shape[0])

        score_ = (accuracy_score(svm.predict(KK),y_train))
        score_val = (accuracy_score(svm.predict(KK_val),y_val))

        np.save("bestsimga_"+(str)(sigma_*100)+"C_"+(str)(C_)+"learn"+(str)((int)(best_learning_rate*10000000)),np.array(loss_history))
        np.save("besttheta_of_simga_"+(str)(sigma_*100)+"C_"+(str)(C_)+"learn"+(str)((int)(best_learning_rate*10000000)),svm.theta)

        send_msg("for sigma_ = "+(str)(sigma_) + " c_ = "+(str)(C_)+
                     ",  when learning rate is "+(str)(best_learning_rate)+
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
            best_svm = svm
            
KK_ = np.vstack([np.ones((X.shape[0],)),X.T]).T
loss_history = best_svm.train(KK_,yy,learning_rate=best_learning_rate,reg=best_C ,num_iters=12000,verbose=True,batch_size=KK.shape[0])
np.save("bestsimga_Full_"+(str)(sigma_*100)+"C_"+(str)(C_)+"learn"+(str)((int)(best_learning_rate*10000000)),np.array(loss_history))
score_test = (accuracy_score(svm.predict(KK_test),y_test))
send_msg("done,test score = "+(str)(score_test)+" when c = "+(str)(best_C))

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
