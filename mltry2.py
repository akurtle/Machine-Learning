import pandas as pd
import numpy as np
import sklearn
from sklearn import *

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data= pd.read_csv("C:/Users/yousu/Projects/Python/ml/machine.csv",sep=",")

data.columns=["vendor","model","MYCT","MMIN","MMAX","CACH","CHMIN","CHMAX","PRP","ERP"]

data=data[["MYCT","MMIN","MMAX","CACH","CHMIN","CHMAX","PRP","ERP"]]

predict="ERP"

x=np.array(data.drop([predict],1))

y=np.array(data[predict])

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)

# linear=linear_model.LinearRegression()

# linear.fit(x_train,y_train)

# acc=linear.score(x_test,y_test)

# if(acc>0.98):
#     with open("C:/Users/yousu/Projects/Python/ml/model.pickle","wb") as f:
#         pickle.dump(linear,f)
# else:
#     while(acc<0.98):
        
#         x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)

#         linear.fit(x_train,y_train)

#         acc=linear.score(x_test,y_test)
#         print(acc)
#         with open("C:/Users/yousu/Projects/Python/ml/model.pickle","wb") as f:
#             pickle.dump(linear,f)


file = open("C:/Users/yousu/Projects/Python/ml/model.pickle","rb")

linear=pickle.load(file)


linear.fit(x_train,y_train)

acc=linear.score(x_test,y_test)

print(acc)

predictions=linear.predict(x_test)


for i in range(len(predictions)):
    print(predictions[i],x_test[i],y_test[i])

# p="MYCT"
# style.use("ggplot")
# pyplot.scatter(data[p],data[predict])
# pyplot.xlabel(p)
# pyplot.ylabel("Final Grade")
# pyplot.show()