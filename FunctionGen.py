import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

def gaussian(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-np.power(x - mu,2)/(2*np.power(sigma,2)))

def lorentzian(x,mu,sigma):
    return (1/np.pi)*(np.power(sigma,1))/(np.power(x-mu,2)+np.power(sigma,2))

a=5000
x=[]
x = np.linspace(0, 2*np.pi, a)
cosSqX=np.power(np.cos(x),2)
sinX=np.sin(x)



c=[0.01,0.05,0.1,0.166666667,0.2,0.25,0.333333333,0.5,0.666666667,0.909090909,1,1.25,1.428571429,1.666666667,2]
#c=[0.1,0.01]
gaussianOP=np.empty_like(x)
lorentzianOP=np.empty_like(x)
mu = [0,np.pi,2*np.pi]
for sigma in c:
    g=gaussian(x, mu[0], sigma)+gaussian(x, mu[1], sigma)+gaussian(x, mu[2], sigma)
    l=lorentzian(x,mu[0],sigma)+lorentzian(x,mu[1],sigma)+lorentzian(x,mu[2],sigma)
    gaussianOP = np.vstack((gaussianOP,g))
    lorentzianOP= np.vstack((lorentzianOP,l))
plotODF = False
if plotODF:
    plt.plot(x,gaussianOP[1::,:].T,label='Gaussian')
    plt.plot(x,lorentzianOP[1::,:].T,'-.',label='Lorentzian')
    plt.legend()
    plt.xlabel(r'$\theta$ (radians)')
    plt.ylabel(r'I($\theta$) (a.u.)')
    plt.show()
    plt.close()

P2g=[]
P2l=[]
T2g=[]
T2l=[]
angle1=[]
angle2=[]
angle3=[]
angle4=[]

for i in range(1,np.shape(gaussianOP)[0]):
    num3DG=gaussianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)]
    den3DG=gaussianOP[i,0:int(a/2)]*sinX[0:int(a/2)]
    cosSq3DG=trapz(num3DG,x[0:int(a/2)])/trapz(den3DG,x[0:int(a/2)])
    #cosSq3DG=np.sum(gaussianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gaussianOP[i,0:int(a/2)]*sinX[0:int(a/2)])
    P2g.append(np.round(1.5*cosSq3DG-0.5,2))
    num3DL=lorentzianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)]
    den3DL=lorentzianOP[i,0:int(a/2)]*sinX[0:int(a/2)]
    cosSq3DL=trapz(num3DL,x[0:int(a/2)])/trapz(den3DL,x[0:int(a/2)])
    #approximation integration as summation - difference in 4th decimal value
    #cosSq3DL1=np.sum(lorentzianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)]*sinX[0:int(a/2)])
    #print(cosSq3DL,cosSq3DL1)
    P2l.append(np.round(1.5*cosSq3DL-0.5,2))    
    cosSq2DG=np.sum(gaussianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gaussianOP[i,0:int(a/2)])
    T2g.append(np.round(2*cosSq2DG-1,2))
    cosSq2DL=np.sum(lorentzianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)])
    T2l.append(np.round(2*cosSq2DL-1,2))

##    angle1.append(180*np.arccos(np.sqrt(cosSq3DG))/3.14)
##    angle2.append(180*np.arccos(np.sqrt(cosSq3DL))/3.14)
##    angle3.append(180*np.arccos(np.sqrt(cosSq2DG))/3.14)
##    angle4.append(180*np.arccos(np.sqrt(cosSq2DL))/3.14)
    
    
##print(P2g,'\n',P2l,'\n',T2g,'\n',T2l)
##print(angle1, angle2, angle3, angle4)



gaussianOP1=np.empty_like(x)
lorentzianOP1=np.empty_like(x)

mu1=[np.pi/2,3*np.pi/2]
for sigma in c:
    g1=gaussian(x, mu1[0], sigma)+gaussian(x, mu1[1], sigma)
    l1=lorentzian(x,mu1[0],sigma)+lorentzian(x,mu1[1],sigma)
    gaussianOP1 = np.vstack((gaussianOP1,g1))
    lorentzianOP1= np.vstack((lorentzianOP1,l1))

if plotODF:
    plt.plot(x,gaussianOP1[1::,:].T,label='Gaussian')
    plt.plot(x,lorentzianOP1[1::,:].T,'-.',label='Lorentzian')
    plt.legend()
    plt.xlabel(r'$\theta$ (radians)')
    plt.ylabel(r'I($\theta$) (a.u.)')
    plt.show()
    plt.close()

P2g1=[]
P2l1=[]
T2g1=[]
T2l1=[]
for i in range(1,np.shape(gaussianOP)[0]):
    cosSq3DG1=np.sum(gaussianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gaussianOP1[i,0:int(a/2)]*sinX[0:int(a/2)])
    P2g1.append(np.round(1.5*cosSq3DG1-0.5,2))
    cosSq3DL1=np.sum(lorentzianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(lorentzianOP1[i,0:int(a/2)]*sinX[0:int(a/2)])
    P2l1.append(np.round(1.5*cosSq3DL1-0.5,2))    
    cosSq2DG1=np.sum(gaussianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gaussianOP1[i,0:int(a/2)])
    T2g1.append(np.round(2*cosSq2DG1-1,2))
    cosSq2DL1=np.sum(lorentzianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(lorentzianOP1[i,0:int(a/2)])
    T2l1.append(np.round(2*cosSq2DL1-1,2))
    
#print(P2g1,'\n',P2l1,'\n',T2g1,'\n',T2l1)







plt.plot(c,P2g,'r-*')
plt.plot(c,P2l,'r--*')
plt.plot(c,T2g,'b-o',)
plt.plot(c,T2l,'b--o')

plt.plot(c,P2g1,'r-*',label='P2 gaussian')
plt.plot(c,P2l1,'r--*',label='P2 lorentzian')
plt.plot(c,T2g1,'b-o',label='T2 gaussian')
plt.plot(c,T2l1,'b--o', label='T2 lorentzian')
plt.legend()
plt.xlabel('Half Width (a.u.)')
plt.ylabel('Orientation Parameter')
plt.show()
plt.close()

##plt.plot(angle1,P2g,'-o')
##plt.plot(angle2,P2l,'-*')
##plt.plot(angle3,T2g,'-+')
##plt.plot(angle4,T2l,'-x')
##plt.show()

