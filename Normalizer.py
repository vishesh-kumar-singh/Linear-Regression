import numpy as np

class normalizer:
    def __init__(self):
        self.denominator=None
        self.min=None
    def normalize(self,X):
        column=X.shape[1]

        min=[]
        for i in range(column):
            val=X[:,i].min()
            min.append(val)
        min_array=np.array(min)

        max=[]
        for i in range(column):
            val=X[:,i].max()
            max.append(val)
        max_array=np.array(max)


        for i in range(len(min_array)):
            if min_array[i]==max_array[i]:
                min_array[i]=0
                max_array[i]=1
        self.min=min_array
        self.denominator=max_array-min_array
        return (X-min_array)/self.denominator
    def spnormalize(self,X,multiple,intercept):
        return (X-intercept)/multiple