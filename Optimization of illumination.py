# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:34:44 2021

@author: John
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import math

p=np.ones(shape=(10,1))
l=np.ones(shape=(625,1))
ldes=np.ones(shape=(625,1))
A=np.zeros(shape=(625,10))
d=np.zeros(shape=(625,10))

#Coordinates of the lamps
lamp=np.array([[4.1,20.4,4],[14.1,21.3,3.5],[22.6,17.1,6],[5.5,12.3,4],[12.2,9.7,4],[15.3,13.8,6],[21.3,10.5,5.5],[3.9,3.3,5],[13.1,4.3,5],[20.3,4.2,4.5]])

#Creating an array that represents the coordinates of the 25x25 pixels
coord=np.indices((25,25))
#Adding 0.5 since we want the coordinates of the center of the pixels
coord=coord+(1/2)

#Calculating the 3D distance between the center of each pixel and the 10 lamps
#and storing them in the array - d[625,10]
#the columns of d represent each one of the lamps
#the 625 rows are coordinates of each pixel
#for example d[24,0] is the distance of lamp 1 from the bottom right square x=24,5,y=0,5(or i=0,j=25)
#similarly d[25,0] represents the pixel x=0.5, y=1.5 (or i=1,j=0) and so on.

for g in range(10):
 z=0
 for i in range (25):
    for j in range(25):
        d[z,g]=math.sqrt(((lamp[g,0]-coord[0,i,j])**2)+(lamp[g,1]-coord[1,i,j])**2+(lamp[g,2]-0)**2)
        z=z+1
             
print('\n','3D distance between the center of the pixels and the 10 lamps','\n','\n',d)

##Calculating the matrix A

#A is proportional to d^(-2), however since the matrix A is scaled so
#that when all lamps have power one, the average illumination level is one
#we assume that A=x*(d^(-2))
#Therefore since Avg(l)=1 we solve Avg(l)=x*(d^(-2))*p for p=1 to find x 
#which gives us x=625/sum(a*p)
a=np.power(d,-2)
x=np.matmul(a,p)
x=625/(sum(x))

A=x*a 
l=np.matmul(A,p) # illumination level for p=1
print('\n','Average illumination level for p=1 is,',np.average(l))

MSE=np.square(np.subtract(1,l)).mean()
RMSE=math.sqrt(MSE)
print('\n','RMS error for p=1 is ',RMSE)

#using least squares to find optimal p we have p=((A.T*A)^(-1))*A.T*ldes

c=np.linalg.pinv(A) #calculating the pseudo inverse of A
p_opt=np.matmul(c,ldes) 
print('\n','Optimal power levels','\n',p_opt)

ldes=np.matmul(A,p_opt) # calculate l optimum for p optimum
MSE2=np.square(np.subtract(1,ldes)).mean()
RMSE2=math.sqrt(MSE2)
print('RMS error for p optimum is ',RMSE2)


#Coordinates of the lamps to be used for the below diagrams
x=[4.1,14.1,22.6,5.5,12.2,15.3,21.3,3.9,13.1,20.3]
y=[20.4,21.3,17.1,12.3,9.7,13.8,10.5,3.3,4.3,4.2]
z=[4.0,3.5,6.0,4.0,4.0,6.0,5.5,5.0,5.0,4.5]

#Reshaping l so it can be plotted
l2=l.reshape(25,25)
l2=l2.T


####Figure 12.5
im1=plt.scatter(x,y,s=15,c='none',edgecolor='black')
for i in range(10):
   plt.annotate(i+1,(x[i],y[i]),xytext=(x[i]+1,y[i]+1),ha='right')
   plt.annotate(z[i],(x[i],y[i]),xytext=(x[i]+2.3,y[i]+1),ha='center')
   plt.annotate('(     m)',(x[i],y[i]),xytext=(x[i]+2.8,y[i]+1.1),ha='center')
   
   
im2=plt.imshow(l2,cmap='turbo',origin='lower',interpolation='none',alpha=1)
x_ticks=[0,24]
x_labels=['0','25 m']
plt.xticks(ticks=x_ticks,labels=x_labels)
y_ticks=[0,24]
y_labels=['0','25 m']
plt.yticks(ticks=y_ticks,labels=y_labels)
plt.clim(0.5,1.5) #setting the min/max values of the colorbar
cbar=plt.colorbar()
cbar.set_ticks([0.6,0.8,1.0,1.2,1.4])
plt.show()

#Reshaping ldes so it can be plotted
ldes2=ldes.reshape(25,25)
ldes2=ldes2.T

im1=plt.scatter(x,y,s=15,c='none',edgecolor='black')
for i in range(10):
   plt.annotate(i+1,(x[i],y[i]),xytext=(x[i]+1,y[i]+1),ha='right')
   plt.annotate(z[i],(x[i],y[i]),xytext=(x[i]+2.3,y[i]+1),ha='center')
   plt.annotate('(     m)',(x[i],y[i]),xytext=(x[i]+2.8,y[i]+1.1),ha='center')
   
   
im2=plt.imshow(ldes2,cmap='turbo',origin='lower',interpolation='none',alpha=1)
x_ticks=[0,24]
x_labels=['0','25 m']
plt.xticks(ticks=x_ticks,labels=x_labels)
y_ticks=[0,24]
y_labels=['0','25 m']
plt.yticks(ticks=y_ticks,labels=y_labels)

cbar2=plt.colorbar()
plt.clim(0.5,1.5)
cbar2.set_ticks([0.6,0.8,1.0,1.2,1.4])
plt.show()

#Figure 12.6
print('\n','Histogram of pixel illumination values for p=1')
plt.hist(l,bins=25,color='mediumpurple',ec='blue')
plt.xlim(0.2,1.8)
plt.ylim(0,120)
plt.xlabel('Intensity')
plt.ylabel('Number of pixels')
plt.show()

print('\n','Histogram of pixel illumination values for p optimum')
plt.hist(ldes,bins=16,color='mediumpurple',ec='blue')
plt.xlim(0.2,1.8)
plt.ylim(0,120)
plt.xlabel('Intensity')
plt.ylabel('Number of pixels')
plt.show()
