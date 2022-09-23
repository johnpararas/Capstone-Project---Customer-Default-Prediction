# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:44:51 2021

@author: John
"""



import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
df=pd.read_excel('.spyder-py3/housedata.xls')
train=df.sample(n=695,replace=False)#getting a random sample of 695 rows
test=df.drop(train.index)#using the rest of the data for testing

y_train=train.iloc[:,5]#selling price
b=np.zeros(shape=(1,3))#b1=u 
A_train=np.ones(shape=(695,3))#for this example we have f1(x)=x,f2(x)=x
A_train=pd.DataFrame(A_train)
for i in range(695):
 A_train.iloc[i,1]=train.iloc[i,3]#area of each house
 A_train.iloc[i,2]=train.iloc[i,1]#number of bedrooms

    ##TASK 1 ##

pinv=np.linalg.pinv(A_train)#pseudo inverse
b=np.matmul(pinv,y_train)
y=np.matmul(A_train,b)

#Regresion Model with 2 basis#
def f(p,l): 
    y=b[1]*p+b[2]*l+b[0]
    return y

y_predicted=np.zeros(shape=(695))
for i in range(695):
    y_predicted[i]=f(train.iloc[i,3],train.iloc[i,1])#predicted prices of houses

     ##TASK 2 ##

pred_price=np.zeros(shape=(774))#calculating the predicted prices of all 774 houses
for i in range(774):
   pred_price[i]=f(df.iloc[i,3],df.iloc[i,1]) 
plt.xlabel('Actual sales prices')
plt.ylabel('Predicted sales prices')
plt.scatter(df.iloc[:,5],pred_price[:],s=10,c='b')
plt.plot(df.iloc[:,5],df.iloc[:,5],c='r')
plt.grid(True)
plt.show()

   #####TASK 3

house=[[1,846,1,115000],[2,1324,2,234500],[3,1150,3,198000],[4,3037,4,528000],[5,3984,5,572500]]
house=pd.DataFrame(house)

house_predicted=np.zeros(shape=(5))
for i in range(5):
    house_predicted[i]=(f(house.iloc[i,1],house.iloc[i,2]))
    
MSE_house=np.square(np.subtract(house_predicted,house.iloc[:,3])).mean()
RMSE_house=math.sqrt(MSE_house)
print('\n','5 Houses data sample RMS error is ',RMSE_house)
    
    ####TASK 4

MSE_train=np.square(np.subtract(y_predicted,train.iloc[:,5])).mean()
RMSE_train=math.sqrt(MSE_train)
print('\n','Training set RMS error is ',RMSE_train)

y_test=np.zeros(shape=(79))
for i in range(79):
    y_test[i]=f(test.iloc[i,3],test.iloc[i,1])

MSE_test=np.square(np.subtract(y_test,test.iloc[:,5])).mean()
RMSE_test=math.sqrt(MSE_test)
print('\n','Test set RMS error is ',RMSE_test)

#Since the errors of the test set and the training set are similar we can
#assume that the model has reasonable generalization ability

      ####TASK 5
x1=train.iloc[:,3]#area
x2=train.iloc[:,1]#bedrooms
x3=train.iloc[:,4]#condo or not
x4=train.iloc[:,0]#location

#f1(x)=1 , f2(x)=x1, f3(x)=max(x1-1500,0)
#f4(x)=x2, f5(x)=x3 , f6,f7,f8 = zip codes

f1=np.ones(shape=695)
f2=x1
f3=np.zeros(shape=695)
for i in range(695):
     f3[i]=max(x1.iloc[i]-1500,0)
f4=x2
f5=x3
f6=np.zeros(shape=(695))
f7=np.zeros(shape=(695))
f8=np.zeros(shape=(695))

for i in range(695):
    if x4.iloc[i]==1:
        f6[i]=0
        f7[i]=0
        f8[i]=0
    elif x4.iloc[i]==2:
        f6[i]=1
        f7[i]=0
        f8[i]=0
    elif x4.iloc[i]==3:
        f6[i]=0
        f7[i]=1
        f8[i]=0
    elif x4.iloc[i]==4:
        f6[i]=0
        f7[i]=0
        f8[i]=1

    ## TASK 6 ##

B_train=np.ones(shape=(695,8))
B_train=pd.DataFrame(B_train,columns=['f1(x)','f2(x)','f3(x)','f4(x)','f5(x)','f6(x)','f7(x)','f8(x)'])
for i in range(695):
 B_train.iloc[i,1]=train.iloc[i,3]
 B_train.iloc[i,2]=f3[i]
 B_train.iloc[i,3]=train.iloc[i,1]
 B_train.iloc[i,4]=train.iloc[i,4]
 B_train.iloc[i,5]=f6[i]
 B_train.iloc[i,6]=f7[i]
 B_train.iloc[i,7]=f8[i]
 
pinv=np.linalg.pinv(B_train)
b=np.matmul(pinv,train.iloc[:,5])

#Regresion Model with 8 basis functions#
def f(x1,x2,x3,x4,x5,x6,x7): 
    y=b[1]*x1+b[2]*x2+b[3]*x3+b[4]*x4+b[5]*x5+b[6]*x6+b[7]*x7+b[0]
    return y

y_predicted2=np.zeros(shape=(695))
for i in range(695):
    y_predicted2[i]=f(train.iloc[i,3],f3[i],train.iloc[i,1],train.iloc[i,4],f6[i],f7[i],f8[i])

MSE_train2=np.square(np.subtract(y_predicted2,train.iloc[:,5])).mean()
RMSE_train2=math.sqrt(MSE_train2)
print('\n','Training set RMS error with 8 basis functions is ',RMSE_train2)

     ## TESTING THE 8 BASIS MODEL ##
x1_test=test.iloc[:,3]#area
x4_test=test.iloc[:,0]
f6_test=np.zeros(shape=(79))
f7_test=np.zeros(shape=(79))
f8_test=np.zeros(shape=(79))
f3_test=np.zeros(shape=(79))
for i in range(79):
    if x4_test.iloc[i]==1:
        f6_test[i]=0
        f7_test[i]=0
        f8_test[i]=0
    elif x4_test.iloc[i]==2:
        f6_test[i]=1
        f7_test[i]=0
        f8_test[i]=0
    elif x4_test.iloc[i]==3:
        f6_test[i]=0
        f7_test[i]=1
        f8_test[i]=0
    elif x4_test.iloc[i]==4:
        f6_test[i]=0
        f7_test[i]=0
        f8_test[i]=1


for i in range(79):
     f3_test[i]=max(x1_test.iloc[i]-1500,0)
y_test=np.zeros(shape=(79))
for i in range(79):
    y_test[i]=f(test.iloc[i,3],f3_test[i],test.iloc[i,1],test.iloc[i,4],f6_test[i],f7_test[i],f8_test[i])

MSE_test2=np.square(np.subtract(y_test,test.iloc[:,5])).mean()
RMSE_test2=math.sqrt(MSE_test2)
print('\n','Test set RMS error with 8 basis functions is ',RMSE_test2)

  ## Scatter plot of actual and predicted prices ##
x1=df.iloc[:,3]#area
x2=df.iloc[:,1]#bedrooms
x3=df.iloc[:,4]#condo or not
x4=df.iloc[:,0]#location

f1=np.ones(shape=774)
f2=x1
f3=np.zeros(shape=774)
for i in range(774):
     f3[i]=max(x1.iloc[i]-1500,0)
f4=x2
f5=x3
f6=np.zeros(shape=(774))
f7=np.zeros(shape=(774))
f8=np.zeros(shape=(774))
for i in range (774):
    if x4.iloc[i]==1:
        f6[i]=0
        f7[i]=0
        f8[i]=0
    elif x4.iloc[i]==2:
        f6[i]=1
        f7[i]=0
        f8[i]=0
    elif x4.iloc[i]==3:
        f6[i]=0
        f7[i]=1
        f8[i]=0
    elif x4.iloc[i]==4:
        f6[i]=0
        f7[i]=0
        f8[i]=1

predicted_price=np.zeros(shape=774)
for i in range(774):
   predicted_price[i]=f(df.iloc[i,3],f3[i],df.iloc[i,1],df.iloc[i,4],f6[i],f7[i],f8[i])

plt.xlabel('Actual sales prices')
plt.ylabel('Predicted sales prices')
plt.scatter(df.iloc[:,5],predicted_price[:],s=10,c='g')
plt.plot(df.iloc[:,5],df.iloc[:,5],c='r')
plt.grid(True)
plt.show()

     ## TASK 7 CROSS VALIDATION ##
 
 ##774/6=129 ,therefore we will have 6 folds and the test set size will be 129 and the rest 645##
 
 #The below functions calculate the RMS train and test error for a given training and test set#
 # Creating a function for the 2 basis model #
def basis_2():
 y_train=train.iloc[:,5]  
 b=np.zeros(shape=(1,3))#b1=u 
 A_train=np.ones(shape=(645,3))
 A_train=pd.DataFrame(A_train)
 for i in range(645):
  A_train.iloc[i,1]=train.iloc[i,3]
  A_train.iloc[i,2]=train.iloc[i,1]

 pinv=np.linalg.pinv(A_train)
 b=np.matmul(pinv,y_train)
 y=np.matmul(A_train,b)

 def f(p,l): #Regresion Model#
    y=b[1]*p+b[2]*l+b[0]
    return y

 y_predicted=np.zeros(shape=(645))
 for i in range(645):
    y_predicted[i]=f(train.iloc[i,3],train.iloc[i,1])
 MSE_train=np.square(np.subtract(y_predicted,train.iloc[:,5])).mean()
 RMSE_train=math.sqrt(MSE_train)
 y_test=np.zeros(shape=(129))
 for i in range(129):
     y_test[i]=f(test.iloc[i,3],test.iloc[i,1])

 MSE_test=np.square(np.subtract(y_test,test.iloc[:,5])).mean()
 RMSE_test=math.sqrt(MSE_test)
 
 return RMSE_train,RMSE_test
 
   ##Creating a function for the 8 basis model##
def basis_8():
 x1=train.iloc[:,3]#area
 x2=train.iloc[:,1]#bedrooms
 x3=train.iloc[:,4]#condo or not
 x4=train.iloc[:,0]#location
 f1=np.ones(shape=645)
 f2=x1
 f3=np.zeros(shape=645)
 for i in range(645):
     f3[i]=max(x1.iloc[i]-1500,0)
 f4=x2
 f5=x3
 f6=np.zeros(shape=(645))
 f7=np.zeros(shape=(645))
 f8=np.zeros(shape=(645))
 for i in range(645):
    if x4.iloc[i]==1:
        f6[i]=0
        f7[i]=0
        f8[i]=0
    elif x4.iloc[i]==2:
        f6[i]=1
        f7[i]=0
        f8[i]=0
    elif x4.iloc[i]==3:
        f6[i]=0
        f7[i]=1
        f8[i]=0
    elif x4.iloc[i]==4:
        f6[i]=0
        f7[i]=0
        f8[i]=1

 B_train=np.ones(shape=(645,8))
 B_train=pd.DataFrame(B_train,columns=['f1(x)','f2(x)','f3(x)','f4(x)','f5(x)','f6(x)','f7(x)','f8(x)'])
 for i in range(645):
  B_train.iloc[i,1]=train.iloc[i,3]
  B_train.iloc[i,2]=f3[i]
  B_train.iloc[i,3]=train.iloc[i,1]
  B_train.iloc[i,4]=train.iloc[i,4]
  B_train.iloc[i,5]=f6[i]
  B_train.iloc[i,6]=f7[i]
  B_train.iloc[i,7]=f8[i]
 
  pinv=np.linalg.pinv(B_train)
  b=np.matmul(pinv,train.iloc[:,5])

 def f(x1,x2,x3,x4,x5,x6,x7): #Regresion Model with 8 basis#
    y=b[1]*x1+b[2]*x2+b[3]*x3+b[4]*x4+b[5]*x5+b[6]*x6+b[7]*x7+b[0]
    return y

 y_predicted2=np.zeros(shape=(645))
 for i in range(645):
    y_predicted2[i]=f(train.iloc[i,3],f3[i],train.iloc[i,1],train.iloc[i,4],f6[i],f7[i],f8[i])

 MSE_train2=np.square(np.subtract(y_predicted2,train.iloc[:,5])).mean()
 RMSE_train2=math.sqrt(MSE_train2)

 x1_test=test.iloc[:,3]
 x4_test=test.iloc[:,0]
 f6_test=np.zeros(shape=(129))
 f7_test=np.zeros(shape=(129))
 f8_test=np.zeros(shape=(129))
 f3_test=np.zeros(shape=(129))
 for i in range(129):
    if x4_test.iloc[i]==1:
        f6_test[i]=0
        f7_test[i]=0
        f8_test[i]=0
    elif x4_test.iloc[i]==2:
        f6_test[i]=1
        f7_test[i]=0
        f8_test[i]=0
    elif x4_test.iloc[i]==3:
        f6_test[i]=0
        f7_test[i]=1
        f8_test[i]=0
    elif x4_test.iloc[i]==4:
        f6_test[i]=0
        f7_test[i]=0
        f8_test[i]=1

 for i in range(129):
     f3_test[i]=max(x1_test.iloc[i]-1500,0)
 y_test=np.zeros(shape=(129))
 for i in range(129):
    y_test[i]=f(test.iloc[i,3],f3_test[i],test.iloc[i,1],test.iloc[i,4],f6_test[i],f7_test[i],f8_test[i])

 MSE_test2=np.square(np.subtract(y_test,test.iloc[:,5])).mean()
 RMSE_test2=math.sqrt(MSE_test2)
 return RMSE_train2,RMSE_test2


data=df.sample(n=774,replace=False)#shuffles the data so that the folds will be random
fold=np.array_split(data,6) # splitting the data in to 6 folds
test=fold[0]#using the first fold data for testing
train=df.drop(test.index)#and the rest of the data for training
fold_error_2_basis=np.zeros(6)
fold_error_8_basis=np.zeros(6)
train_fold_error_2_basis=np.zeros(6)
train_fold_error_8_basis=np.zeros(6)
g=1
for i in range(6):#calculating the error for each fold
    if g ==1:
     y=basis_2()
     z=basis_8()
     fold_error_2_basis[i]=y[1]
     fold_error_8_basis[i]=z[1]
     train_fold_error_2_basis[i]=y[0]
     train_fold_error_8_basis[i]=z[0]
     g=g+1
    elif g==2:
     test=fold[1]
     train=df.drop(test.index)  
     y=basis_2()
     z=basis_8()
     fold_error_2_basis[i]=y[1]
     fold_error_8_basis[i]=z[1]
     train_fold_error_2_basis[i]=y[0]
     train_fold_error_8_basis[i]=z[0]     
     g=g+1
    elif g==3: 
     test=fold[2]
     train=df.drop(test.index)   
     y=basis_2()
     z=basis_8()
     fold_error_2_basis[i]=y[1]
     fold_error_8_basis[i]=z[1]
     train_fold_error_2_basis[i]=y[0]
     train_fold_error_8_basis[i]=z[0]     
     g=g+1
    elif g==4: 
     test=fold[3]
     train=df.drop(test.index) 
     y=basis_2()
     z=basis_8()
     fold_error_2_basis[i]=y[1]
     fold_error_8_basis[i]=z[1]
     train_fold_error_2_basis[i]=y[0]
     train_fold_error_8_basis[i]=z[0]     
     g=g+1
    elif g==5: 
     test=fold[4]
     train=df.drop(test.index)   
     y=basis_2()
     z=basis_8()
     fold_error_2_basis[i]=y[1]
     fold_error_8_basis[i]=z[1]
     train_fold_error_2_basis[i]=y[0]
     train_fold_error_8_basis[i]=z[0]    
     g=g+1
    elif g==6: 
     test=fold[5]
     train=df.drop(test.index)  
     y=basis_2()
     z=basis_8()
     fold_error_2_basis[i]=y[1]
     fold_error_8_basis[i]=z[1]
     train_fold_error_2_basis[i]=y[0]
     train_fold_error_8_basis[i]=z[0]     
     g=g+1

print('\n')
for i in range(6):
 print('Fold',i+1,'TRAINING error for 2 basis:',train_fold_error_2_basis[i],'and TEST error for 2 basis:',fold_error_2_basis[i])
print('\n')
for i in range(6):
 print('Fold',i+1,'TRAINING error for 8 basis:',train_fold_error_8_basis[i],'and TEST error for 8 basis:',fold_error_8_basis[i]) 
    
final_MSE_test_2_basis=np.square(fold_error_2_basis).mean()
final_RMSE_test_2_basis=math.sqrt(final_MSE_test_2_basis)
print('\n','RMS Cross Validation error for 2 basis functions is:',final_RMSE_test_2_basis)
final_MSE_test_8_basis=np.square(fold_error_8_basis).mean()
final_RMSE_test_8_basis=math.sqrt(final_MSE_test_8_basis)
print('RMS Cross Validation error for 8 basis functions is:',final_RMSE_test_8_basis)

#Since the cross validation error for both the 2 basis model and the 8 basis model , is 
#close enough to the training set error we can conclude that the model has 
#reasonable generalization ability.