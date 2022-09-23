# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:29:05 2021

@author: John
"""

#NOTE: The below program needed around 20 minutes to run on the computer it was tested.##

  #####TASK 1####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

digit=pd.read_excel('data.xlsx',sheet_name='dzip',header=None)
train_images=pd.read_excel('data.xlsx',sheet_name='azip',header=None)
digit_test=pd.read_excel('data.xlsx',sheet_name='dtest',header=None)
test_images=pd.read_excel('data.xlsx',sheet_name='testzip',header=None)

def ima(A):
    a1=np.squeeze(A)
    a1=pd.DataFrame(a1)
    a1=a1.values.reshape(16,16)
    a1=preprocessing.normalize(a1)
    return plt.imshow(a1,cmap='Greys')

zeroes=np.zeros(shape=(256,0))
ones=np.zeros(shape=(256,0))
twos=np.zeros(shape=(256,0))
threes=np.zeros(shape=(256,0))
fours=np.zeros(shape=(256,0))
fives=np.zeros(shape=(256,0))
sixes=np.zeros(shape=(256,0))
sevens=np.zeros(shape=(256,0))
eights=np.zeros(shape=(256,0))
nines=np.zeros(shape=(256,0))

#Sorting the digits in classes, each containing the images of a specific digit
for i in digit:
   if digit.iloc[0,i]==0:
        zeroes=np.column_stack((zeroes,train_images.iloc[:,i]))
   elif digit.iloc[0,i]==1:
       ones=np.column_stack((ones,train_images.iloc[:,i]))
   elif digit.iloc[0,i]==2:
       twos=np.column_stack((twos,train_images.iloc[:,i]))
   elif digit.iloc[0,i]==3:
       threes=np.column_stack((threes,train_images.iloc[:,i]))
   elif digit.iloc[0,i]==4:
       fours=np.column_stack((fours,train_images.iloc[:,i]))
   elif digit.iloc[0,i]==5:
       fives=np.column_stack((fives,train_images.iloc[:,i]))
   elif digit.iloc[0,i]==6:
       sixes=np.column_stack((sixes,train_images.iloc[:,i]))
   elif digit.iloc[0,i]==7:
       sevens=np.column_stack((sevens,train_images.iloc[:,i]))
   elif digit.iloc[0,i]==8:
       eights=np.column_stack((eights,train_images.iloc[:,i]))
   elif digit.iloc[0,i]==9:
       nines=np.column_stack((nines,train_images.iloc[:,i]))
       
#Returns the singular images of digit 'a' for 'i' number of basis        
def u_svd(a,i):
 u,s,v=np.linalg.svd(a)
 return u[:,:i]

#Compares one type of digit with a given number of basis images to a test image
def residual(digits,basis,testimg):
   residual_error=np.linalg.norm((np.identity(256)-u_svd(digits,basis)[:,:10]@u_svd(digits,basis)[:,:10].T)@test_images.iloc[:,testimg] ,2)
   relative_error=residual_error/np.linalg.norm(test_images.iloc[:,0] ,2)
   return relative_error

error_k_5=np.zeros(shape=(10,2007))#rows are digits and columns are the errors for each digit
classifier_k_5=np.zeros(shape=(10,1))#contains the number of classified digits for each class
error_k_8=np.zeros(shape=(10,2007))
classifier_k_8=np.zeros(shape=(10,1))
error_k_10=np.zeros(shape=(10,2007))
classifier_k_10=np.zeros(shape=(10,1))
error_k_20=np.zeros(shape=(10,2007))
classifier_k_20=np.zeros(shape=(10,1))
accuracy_k_5=np.zeros(shape=(10,1))#contains the number of correctly classified digits for each class
accuracy_k_8=np.zeros(shape=(10,1))
accuracy_k_10=np.zeros(shape=(10,1))
accuracy_k_20=np.zeros(shape=(10,1))
error_k_mix=np.zeros(shape=(10,2007))
classifier_k_mix=np.zeros(shape=(10,1))
accuracy_k_mix=np.zeros(shape=(10,1))

for i in range(2007):
    error_k_5[0,i]=residual(zeroes,5,i)#5 basis
    error_k_5[1,i]=residual(ones,5,i)
    error_k_5[2,i]=residual(twos,5,i)
    error_k_5[3,i]=residual(threes,5,i)
    error_k_5[4,i]=residual(fours,5,i)
    error_k_5[5,i]=residual(fives,5,i)
    error_k_5[6,i]=residual(sixes,5,i)
    error_k_5[7,i]=residual(sevens,5,i)
    error_k_5[8,i]=residual(eights,5,i)
    error_k_5[9,i]=residual(nines,5,i)
    x_5=np.argmin(error_k_5[:,i])
    classifier_k_5[x_5,0]+=1
    if x_5==digit_test.iloc[0,i]:
        accuracy_k_5[x_5]+=1 # counts how many classified digits are correct for 5 basis
    error_k_8[0,i]=residual(zeroes,8,i)#8 basis
    error_k_8[1,i]=residual(ones,8,i)
    error_k_8[2,i]=residual(twos,8,i)
    error_k_8[3,i]=residual(threes,8,i)
    error_k_8[4,i]=residual(fours,8,i)
    error_k_8[5,i]=residual(fives,8,i)
    error_k_8[6,i]=residual(sixes,8,i)
    error_k_8[7,i]=residual(sevens,8,i)
    error_k_8[8,i]=residual(eights,8,i)
    error_k_8[9,i]=residual(nines,8,i)
    x_8=np.argmin(error_k_8[:,i])
    classifier_k_8[x_8,0]+=1
    if x_8==digit_test.iloc[0,i]:
        accuracy_k_8[x_8]+=1
    error_k_10[0,i]=residual(zeroes,10,i)#10 basis
    error_k_10[1,i]=residual(ones,10,i)
    error_k_10[2,i]=residual(twos,10,i)
    error_k_10[3,i]=residual(threes,10,i)
    error_k_10[4,i]=residual(fours,10,i)
    error_k_10[5,i]=residual(fives,10,i)
    error_k_10[6,i]=residual(sixes,10,i)
    error_k_10[7,i]=residual(sevens,10,i)
    error_k_10[8,i]=residual(eights,10,i)
    error_k_10[9,i]=residual(nines,10,i)
    x_10=np.argmin(error_k_10[:,i])
    classifier_k_10[x_10,0]+=1
    if x_10==digit_test.iloc[0,i]:
        accuracy_k_10[x_10]+=1
    error_k_20[0,i]=residual(zeroes,20,i)#20 basis
    error_k_20[1,i]=residual(ones,20,i)
    error_k_20[2,i]=residual(twos,20,i)
    error_k_20[3,i]=residual(threes,20,i)
    error_k_20[4,i]=residual(fours,20,i)
    error_k_20[5,i]=residual(fives,20,i)
    error_k_20[6,i]=residual(sixes,20,i)
    error_k_20[7,i]=residual(sevens,20,i)
    error_k_20[8,i]=residual(eights,20,i)
    error_k_20[9,i]=residual(nines,20,i)
    x_20=np.argmin(error_k_20[:,i])
    classifier_k_20[x_20,0]+=1
    if x_20==digit_test.iloc[0,i]:
        accuracy_k_20[x_20]+=1
    error_k_mix[0,i]=residual(zeroes,10,i)#mixed basis
    error_k_mix[1,i]=residual(ones,8,i)
    error_k_mix[2,i]=residual(twos,10,i)
    error_k_mix[3,i]=residual(threes,7,i)
    error_k_mix[4,i]=residual(fours,6,i)
    error_k_mix[5,i]=residual(fives,10,i)
    error_k_mix[6,i]=residual(sixes,10,i)
    error_k_mix[7,i]=residual(sevens,10,i)
    error_k_mix[8,i]=residual(eights,10,i)
    error_k_mix[9,i]=residual(nines,10,i)
    x_mix=np.argmin(error_k_mix[:,i])
    classifier_k_mix[x_mix,0]+=1
    if x_mix==digit_test.iloc[0,i]:
        accuracy_k_mix[x_mix]+=1   
        
accuracy_table=np.zeros(shape=(1,4))
row_name=['correct digits (%)']
column_name=['5-basis images','8-basis images','10-basis images','20-basis images']
accuracy_table=pd.DataFrame(accuracy_table,index=row_name,columns=column_name)
accuracy_table.iloc[0,0]=((accuracy_k_5.sum())/2007)*100
accuracy_table.iloc[0,1]=((accuracy_k_8.sum())/2007)*100
accuracy_table.iloc[0,2]=((accuracy_k_10.sum())/2007)*100
accuracy_table.iloc[0,3]=((accuracy_k_20.sum())/2007)*100
pd.set_option('display.max_columns',10)
print('\n')
print('Percentage of correctly classified digits using 5,8,10 and 20 basis images')
print('\n',accuracy_table)

#From the above table it is evident that using more than 10 basis does not reduce the error

   ###TASK 2###
total_error=np.zeros(shape=(1,10))
print('\n')
print('Below is the accuracy % per digit class using 10 basis')
for i in range(10):
    b=np.count_nonzero(digit_test.iloc[0,:]==i)#b is the number of times digit i appears in the test set
    total_error[0,i]=(accuracy_k_10[i]/b)*100
    print('% of correctly classified digit',i,'is',total_error[0,i])
#From the above numbers we can assume that the hardest digits to classify are 2,3,5 and 8
#That is because they have the lowest % of correct classification results.

     ###TASK 3###
def bases(digit,test_digit):    
 bases_error=np.zeros(shape=(11))
 for i in range(11):
     bases_error[i]=residual(digit,i,test_digit)
 k=[0,1,2,3,4,5,6,7,8,9,10]   
 fig1=plt.figure(figsize=(10,4.8))
 ax1=fig1.add_subplot(121)
 plt.xlabel('Basis')
 plt.ylabel('Residual')
 x_ticks=[0,1,2,3,4,5,6,7,8,9,10]
 plt.xticks(ticks=x_ticks)
 plt.plot(k,bases_error)
 ax2=fig1.add_subplot(122)
 img=ima(test_images.iloc[:,test_digit])
 plt.axis('off')
 return plt.show()

print('\n','Below are the approximations using 10 basis for well writen digits')
bases(zeroes,6)#note that the right values are the positions in the test_images matrix
bases(ones,128)
bases(twos,94)
bases(threes,2)
bases(fours,245)
bases(fives,51)
bases(sixes,55)
bases(sevens,86)
bases(eights,50)
bases(nines,0)

#From the above graphs and the accuracy table from #Task 2 we can see that the maximum 
#number of useful basis is 10. After that point using more basis does not result 
#in smaller error values. On the other hand using fewer basis does not make sence since 
#that would result in higher error values and less overall accuracy.
#To prove this , we perform an experiment with mixed number of basis.
#In the table below are the accuracy results for k=10 for all digits except 1,3 and 4 .
#For '1' 8 basis were used, 7 basis for '3' and 6 for '4'. 
#The reason these numbers were chose is because for number '1' according to the above graph 
#(approximations of well writen digits) the error seems to stabilize at 8 basis.
#For similar reasons the other numbers had basis of 6 and 7. 

print('\n','Below are the results of using fewer basis for some numbers')

total_error=np.zeros(shape=(1,10))
for i in range(10):
    b_2=np.count_nonzero(digit_test.iloc[0,:]==i)
    total_error[0,i]=(accuracy_k_mix[i]/b_2)*100
    print('% of correctly classified digit',i,'is',total_error[0,i])

print('\n','The accuracy percentage when using fewer basis for some classes:',(accuracy_k_mix.sum()/2007)*100)

#The accuracy of the algorithm is smaller , as expected.
#Therefore we conclude that the we should use the same number of basis for all numbers 
#which in this case is k=10

