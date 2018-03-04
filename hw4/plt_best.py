import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize = (18,12))
import os
for filename in os.listdir("/Users/xavier.qiu/Documents/GitHub/selfImplementML/hw4"):
    if filename.startswith("bestsimga"):
        print("1")
        loss_ = np.load(filename)
        plt.plot(loss_)
plt.ylim(0,0.2)
plt.show()


#%%