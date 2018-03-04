import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.figure(figsize = (18,12))
import os
for filename in os.listdir("/Users/xavier.qiu/Documents/GitHub/selfImplementML/hw4"):
    if filename.startswith("bestsimga_44"):
        loss_ = np.load(filename)
        plt.plot(loss_, label=filename[17:],alpha = 1/4*loss_[-1])
        
plt.ylim(0,1)
plt.legend()
plt.show()


#%%