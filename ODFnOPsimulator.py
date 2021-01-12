'''This program calculates the Chebyshev/Herman orientation parameters
from simulated intensity distribution data.

Code developed by Dr. A. Kaniyoor,
Macromolecular Materials Laboratory,University of Cambridge, Cambridge, UK
2020-2021

Reference Publication: Quantifying Alignment in Carbon Nanotube Yarns and Similar 2D Anisotropic Systems
A. Kaniyoor, T.S. Gspann, J. E. Mizen, J.A. Elliott.
To be submitted

There are two main programs here. Program 1 (True by default) generates orientation distribution functions with varying widths 
and calculates orientation parameters from the ODFs. To view the ODFs, please enable command - plot ODF=True (False by default)
Program 2 (off/False by default) generates ODFs with secondary peaks whose height can be adjusted in the code, and calculates orientation parameters.

'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from matplotlib import rcParams
rcParams['font.family']='Arial'
rcParams['legend.fontsize']=8 # 11 for 0.5 column figure
rcParams['axes.labelsize']=12 #16 for 0.5 column figure,12 for 1 column figure
rcParams['xtick.labelsize']=10 # 14 for 0.5 column figure
rcParams['ytick.labelsize']=10
rcParams['lines.markersize']=3
rcParams['lines.linewidth']=1
rcParams['lines.antialiased']=True
rcParams['mathtext.default']='regular'
rcParams['figure.figsize']=3.5,3.2 #3.5,2.6 is 1 column figure
rcParams['figure.dpi']=150 # change dpi=300 for publication quality images



# defining the functional forms
def gaussian(x, mu, hwidth):
    sigma=hwidth/(np.sqrt(2*np.log(2)))
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(np.abs(x - mu)/(np.sqrt(2)*sigma))**2)

def lorentzian(x,mu,hwidth):
    return (1/(np.pi*hwidth))*(hwidth**2)/(((x-mu)**2)+hwidth**2)

def gnd(x,mu,hwidth,beta):
    alpha=hwidth/((np.log(2))**(1/beta))
    return (beta/(2*alpha*sp.gamma(1/beta)))*np.exp(-(np.abs(x - mu)/alpha)**beta)



# generating angle data
a=5000
x=[]
x = np.linspace(0, 360, a)
cosSqX=np.power(np.cos(x*np.pi/180),2)
sinX=np.sin(x*np.pi/180)
cosFtX=np.power(np.cos(x*np.pi/180),4)



# Choose the program to run: program 1 - primary peaks only ; prgoram 2 - with secondary peaks

program1 = True
program2 = False





###### PROGRAM 1 for generating ODFs with primary peaks #####


if program1:
    
    # setting peak parameters
    c=[0.01,0.6,2.9,5.7,9.5,11.5,14.3,19.1,28.6,38.2,52.1,57.3,71.6,81.8,95.5,114.6,120,135,150,165,180]
    #c=[5]
    beta=1.5
    mu = [0,180,360]
    mu1=[-90,90,270,450]

    
    plotODF = False # True only if you want to view ODFs, usually when c is a fixed value

    
    # generating ODFs oriented along reference direction
    gaussianOP=np.empty_like(x)
    lorentzianOP=np.empty_like(x)
    gndOP=np.empty_like(x)
    for hwidth in c:
        g=gaussian(x, mu[0], hwidth)+gaussian(x, mu[1],hwidth)+gaussian(x, mu[2],hwidth)
        l=lorentzian(x,mu[0],hwidth)+lorentzian(x,mu[1],hwidth)+lorentzian(x,mu[2],hwidth)
        gn=gnd(x,mu[0],hwidth,beta)+gnd(x,mu[1],hwidth,beta)+gnd(x,mu[2],hwidth,beta)
        gaussianOP = np.vstack((gaussianOP,g))
        lorentzianOP= np.vstack((lorentzianOP,l))
        gndOP=np.vstack((gndOP,gn))

    #output= np.array([x,l])
    #np.savetxt('ModelData.txt',output.T)

    # generating ODFs oriented perpendicular to reference direction   
    gaussianOP1=np.empty_like(x)
    lorentzianOP1=np.empty_like(x)
    gndOP1=np.empty_like(x)

    for hwidth in c:
        g1=gaussian(x, mu1[0],hwidth)+gaussian(x, mu1[1],hwidth)+gaussian(x, mu1[2],hwidth)+gaussian(x, mu1[3],hwidth)
        l1=lorentzian(x,mu1[0],hwidth)+lorentzian(x,mu1[1],hwidth)+lorentzian(x,mu1[2],hwidth)+lorentzian(x,mu1[3],hwidth)
        gn1=gnd(x, mu1[0], hwidth,beta)+gnd(x, mu1[1], hwidth,beta)++gnd(x, mu1[2], hwidth,beta)++gnd(x, mu1[3], hwidth,beta)
        gaussianOP1 = np.vstack((gaussianOP1,g1))
        lorentzianOP1= np.vstack((lorentzianOP1,l1))
        gndOP1=np.vstack((gndOP1,gn1))
        

    # plotting ODF
    if plotODF:
        plt.figure(figsize=(3.5,3))
        plt.plot(x,lorentzianOP[1::,:].T,'b-',label='LD')
        plt.plot(x,gndOP[1::,:].T,'g-.',label='GND')
        plt.plot(x,gaussianOP[1::,:].T,'r--',label='GD')
        plt.legend(bbox_to_anchor=(0,1.1,1,0), loc="lower left",mode="expand", ncol=3)
        plt.xlabel('Angle (\xb0)')
        plt.ylabel('Intensity (a.u.)')
        plt.xticks([0,90,180,270,360])
        plt.locator_params('y',nbins=6)
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        plt.tight_layout()
        plt.minorticks_on()
        plt.show()
        plt.close()

        plt.figure(figsize=(3.5,3))
        plt.plot(x,lorentzianOP1[1::,:].T,'b-',label='LD')
        plt.plot(x,gndOP1[1::,:].T,'g-.',label='GND')
        plt.plot(x,gaussianOP1[1::,:].T,'r--',label='GD')
        plt.legend(bbox_to_anchor=(0,1.1,1,0), loc="lower left",mode="expand", ncol=3)
        plt.xlabel('Angle (\xb0)')
        plt.ylabel('Intensity (a.u.)')
        plt.xticks([0,90,180,270,360])
        plt.locator_params('y',nbins=6)
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        plt.tight_layout()
        plt.show()
        plt.close()

    # calculating orientation parameters
    P2g=[]
    P2l=[]
    T2g=[]
    T2l=[]
    P2gn=[]
    T2gn=[]
    T4l=[]

    P2g1=[]
    P2l1=[]
    T2g1=[]
    T2l1=[]
    P2gn1=[]
    T2gn1=[]
    T4l1=[]

    for i in range(1,np.shape(gaussianOP)[0]):
        cosSq3DG=np.sum(gaussianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gaussianOP[i,0:int(a/2)]*sinX[0:int(a/2)])
        P2g.append(np.round(1.5*cosSq3DG-0.5,3))
        cosSq3DL=np.sum(lorentzianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)]*sinX[0:int(a/2)])
        P2l.append(np.round(1.5*cosSq3DL-0.5,3))
        cosSq3DGN=np.sum(gndOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gndOP[i,0:int(a/2)]*sinX[0:int(a/2)])
        P2gn.append(np.round(1.5*cosSq3DGN-0.5,3))
        cosSq2DG=np.sum(gaussianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gaussianOP[i,0:int(a/2)])
        T2g.append(np.round(2*cosSq2DG-1,3))
        cosSq2DL=np.sum(lorentzianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)])
        T2l.append(np.round(2*cosSq2DL-1,3))
        cosSq2DGN=np.sum(gndOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gndOP[i,0:int(a/2)])
        T2gn.append(np.round(2*cosSq2DGN-1,3))
        
        
    for i in range(1,np.shape(gaussianOP)[0]):
        cosSq3DG1=np.sum(gaussianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gaussianOP1[i,0:int(a/2)]*sinX[0:int(a/2)])
        P2g1.append(np.round(1.5*cosSq3DG1-0.5,3))
        cosSq3DL1=np.sum(lorentzianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(lorentzianOP1[i,0:int(a/2)]*sinX[0:int(a/2)])
        P2l1.append(np.round(1.5*cosSq3DL1-0.5,3))
        cosSq3DGN1=np.sum(gndOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gndOP1[i,0:int(a/2)]*sinX[0:int(a/2)])
        P2gn1.append(np.round(1.5*cosSq3DGN1-0.5,3))
        cosSq2DG1=np.sum(gaussianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gaussianOP1[i,0:int(a/2)])
        T2g1.append(np.round(2*cosSq2DG1-1,3))
        cosSq2DL1=np.sum(lorentzianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(lorentzianOP1[i,0:int(a/2)])
        T2l1.append(np.round(2*cosSq2DL1-1,3))
        cosSq2DGN1=np.sum(gndOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gndOP1[i,0:int(a/2)])
        T2gn1.append(np.round(2*cosSq2DGN1-1,3))
        
        
    # Plotting orientation parameters 
    fw=2*np.asarray(c) # converting widths to full widths (FWHM)

    plt.plot(fw,P2l,'b-x', label='<P$_2$>$_{LD}$')
    plt.plot(fw,P2gn,'g--x', label='<P$_2$>$_{GND}$')
    plt.plot(fw,P2g,'r:x', label = '<P$_2$>$_{GD}$')
    plt.plot(fw,P2l1,'b-x')
    plt.plot(fw,P2gn1,'g--x')
    plt.plot(fw,P2g1,'r:x')


    plt.plot(fw,T2l,'b-o', label='<T$_2$>$_{LD}$')
    plt.plot(fw,T2gn,'g--o', label='<T$_2$>$_{GND}$')
    plt.plot(fw,T2g,'r:o', label='<T$_2$>$_{GD}$')
    plt.plot(fw,T2l1,'b-o')
    plt.plot(fw,T2gn1,'g--o')
    plt.plot(fw,T2g1,'r:o')

    plt.legend(bbox_to_anchor=(0,1,1,0), loc="lower left",mode="expand", ncol=3)
    #plt.legend()
    plt.xlabel('FWHM (\xb0)')
    plt.ylabel('Orientation Parameter')
    #plt.title('OP vs FWHM')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
    plt.close()

    print('\n <T2> vs. FWHM')
    print('+LD\n',T2l,'\n-LD\n',T2l1,'\n+GD\n',T2g,'\n-GD\n',T2g1,'\n+GND\n',T2gn,'\n-GND\n',T2gn1)
    print('\n <P2> vs. FWHM')
    print('+LD\n',P2l,'\n-LD\n',P2l1,'\n+GD\n',P2g,'\n-GD\n',P2g1,'\n+GND\n',P2gn,'\n-GND\n',P2gn1)




    '''
    # Plotting T4 values

    for i in range(1,np.shape(gaussianOP)[0]):
        cosFt2DL=np.sum(lorentzianOP[i,0:int(a/2)]*cosFtX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)])
        T4l.append(np.round(8*cosFt2DL-8*cosSq2DL+1,3))
        cosFt2DL1=np.sum(lorentzianOP1[i,0:int(a/2)]*cosFtX[0:int(a/2)])/np.sum(lorentzianOP1[i,0:int(a/2)])
        T4l1.append(np.round(8*cosFt2DL1-8*cosSq2DL1+1,3))
        
    plt.figure()
    plt.plot(fw,T4l,'b-o', label='<T$_4$>$_{LD}$')
    plt.plot(fw,T4l1,'b-o')
    plt.legend()
    plt.xlabel('FWHM (\xb0)')
    plt.ylabel('Orientation Parameter')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
    plt.close()
    print('\n',T4l,T4l1)
    '''
    


    
    # to calculate Orientation parameter for GND function with different shape factors
    b=np.linspace(1,2,5)
    for beta in b:
        gaussianOP=np.empty_like(x)
        lorentzianOP=np.empty_like(x)
        gndOP=np.empty_like(x)
        
        for hwidth in c:
            g=gaussian(x, mu[0], hwidth)+gaussian(x, mu[1],hwidth)+gaussian(x, mu[2],hwidth)
            l=lorentzian(x,mu[0],hwidth)+lorentzian(x,mu[1],hwidth)+lorentzian(x,mu[2],hwidth)
            gn=gnd(x,mu[0],hwidth,beta)+gnd(x,mu[1],hwidth,beta)+gnd(x,mu[2],hwidth,beta)
            gaussianOP = np.vstack((gaussianOP,g))
            lorentzianOP= np.vstack((lorentzianOP,l))
            gndOP=np.vstack((gndOP,gn))
       
    
        P2g=[]
        P2l=[]
        T2g=[]
        T2l=[]
        P2gn=[]
        T2gn=[]
        
        for i in range(1,np.shape(gaussianOP)[0]):
            cosSq3DG=np.sum(gaussianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gaussianOP[i,0:int(a/2)]*sinX[0:int(a/2)])
            P2g.append(np.round(1.5*cosSq3DG-0.5,3))
            cosSq3DL=np.sum(lorentzianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)]*sinX[0:int(a/2)])
            P2l.append(np.round(1.5*cosSq3DL-0.5,3))
            cosSq3DGN=np.sum(gndOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gndOP[i,0:int(a/2)]*sinX[0:int(a/2)])
            P2gn.append(np.round(1.5*cosSq3DGN-0.5,3))
            cosSq2DG=np.sum(gaussianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gaussianOP[i,0:int(a/2)])
            T2g.append(np.round(2*cosSq2DG-1,3))
            cosSq2DL=np.sum(lorentzianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)])
            T2l.append(np.round(2*cosSq2DL-1,3))
            cosSq2DGN=np.sum(gndOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gndOP[i,0:int(a/2)])
            T2gn.append(np.round(2*cosSq2DGN-1,3))
    
        print('\n <T2> vs. FWHM for GNDs with beta=', beta)
        print(T2gn)
        plt.plot(fw,T2gn,'--o',label=(r'$\beta$='+str(beta)))
                
        #print('\n <P2> vs. FWHM for GNDs with beta=', beta)
        #print(P2gn)
        #plt.plot(fw,P2gn,'--o',label=(r'$\beta$='+str(beta)))

        
               
        plotNegSide=False
        if plotNegSide:
            gaussianOP1=np.empty_like(x)
            lorentzianOP1=np.empty_like(x)
            gndOP1=np.empty_like(x)
    
            for hwidth in c:
                g1=gaussian(x, mu1[0],hwidth)+gaussian(x, mu1[1],hwidth)+gaussian(x, mu1[2],hwidth)+gaussian(x, mu1[3],hwidth)
                l1=lorentzian(x,mu1[0],hwidth)+lorentzian(x,mu1[1],hwidth)+lorentzian(x,mu1[2],hwidth)+lorentzian(x,mu1[3],hwidth)
                gn1=gnd(x, mu1[0], hwidth,beta)+gnd(x, mu1[1], hwidth,beta)++gnd(x, mu1[2], hwidth,beta)++gnd(x, mu1[3], hwidth,beta)
                gaussianOP1 = np.vstack((gaussianOP1,g1))
                lorentzianOP1= np.vstack((lorentzianOP1,l1))
                gndOP1=np.vstack((gndOP1,gn1))
                
            P2g1=[]
            P2l1=[]
            T2g1=[]
            T2l1=[]
            P2gn1=[]
            T2gn1=[]
            for i in range(1,np.shape(gaussianOP)[0]):
                cosSq3DG1=np.sum(gaussianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gaussianOP1[i,0:int(a/2)]*sinX[0:int(a/2)])
                P2g1.append(np.round(1.5*cosSq3DG1-0.5,3))
                cosSq3DL1=np.sum(lorentzianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(lorentzianOP1[i,0:int(a/2)]*sinX[0:int(a/2)])
                P2l1.append(np.round(1.5*cosSq3DL1-0.5,3))
                cosSq3DGN1=np.sum(gndOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gndOP1[i,0:int(a/2)]*sinX[0:int(a/2)])
                P2gn1.append(np.round(1.5*cosSq3DGN1-0.5,3))
                cosSq2DG1=np.sum(gaussianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gaussianOP1[i,0:int(a/2)])
                T2g1.append(np.round(2*cosSq2DG1-1,3))
                cosSq2DL1=np.sum(lorentzianOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(lorentzianOP1[i,0:int(a/2)])
                T2l1.append(np.round(2*cosSq2DL1-1,3))
                cosSq2DGN1=np.sum(gndOP1[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gndOP1[i,0:int(a/2)])
                T2gn1.append(np.round(2*cosSq2DGN1-1,3))
            
    
            print('\n <T2> vs. FWHM for GNDs with beta=', beta)
            print(T2gn1)
            plt.plot(fw,T2gn1,'--o',label=(r'$ \beta$='+str(beta)))
            #print('\n <P2> vs. FWHM for GNDs with beta=', beta)
            #print(P2gn1)
            #plt.plot(fw,P2gn1,'--x',label=(r'$ \beta$='+str(beta)))
            
    
    
    
    plt.plot(fw,T2l,'b-x',label='LD')
    plt.plot(fw,T2g,'k-x',label='GD')
    #plt.plot(fw,P2l,'b-x',label='LD')
    #plt.plot(fw,P2g,'k-x',label='GD')
    plt.legend(bbox_to_anchor=(0,1,1,0), loc="lower left",mode="expand", ncol=4)
    plt.xlabel('FWHM (\xb0)')
    plt.ylabel('Orientation Parameter')
    plt.tight_layout()
    plt.minorticks_on()
    plt.show()
    plt.close()
    
    print('\n <T2> vs. FWHM for LD')
    print(T2l)
    print('\n <T2> vs. FWHM for GD')
    print(T2g)
    #print('P2 Lorentzian =',P2l)
    #print('P2 Gaussian =', P2g)

    









###### PROGRAM 2 for generating ODFs with secondary orientation peaks #####
if program2:

    # Setting peak parameters

    c=[0.1,0.6,2.9,5.7,9.5,11.5,14.3,19.1,28.6,38.2,52.1,57.3,71.6,81.8,95.5,114.6,120,135,150,165,180]
    h=[0,0.1,0.25,0.5,0.75,1] # relative height of secondary peak
    #c=[5]
    #h=[0.1,0.75] # do not forget to change labels in legend of ODF plots if you change these values
    beta=1.5
    mu2 = [0,90,180,270,360] # setting peak positions

    plotODF=False

    # defining ODFs
    gaussianOP=np.empty_like(x)
    lorentzianOP=np.empty_like(x)
    gndOP=np.empty_like(x)

    for height in h:
        for hwidth in c:
            g=gaussian(x, mu2[0], hwidth)+height*gaussian(x, mu2[1],hwidth)+gaussian(x, mu2[2],hwidth)+height*gaussian(x, mu2[3],hwidth)+gaussian(x, mu2[4],hwidth)
            l=lorentzian(x,mu2[0],hwidth)+height*lorentzian(x,mu2[1],hwidth)+lorentzian(x,mu2[2],hwidth)+height*lorentzian(x,mu2[3],hwidth)+lorentzian(x,mu2[4],hwidth)
            gn=gnd(x,mu2[0],hwidth,beta)+height*gnd(x,mu2[1],hwidth,beta)+gnd(x,mu2[2],hwidth,beta)+height*gnd(x,mu2[3],hwidth,beta)+gnd(x,mu2[4],hwidth,beta)
            gaussianOP = np.vstack((gaussianOP,g))
            lorentzianOP= np.vstack((lorentzianOP,l))
            gndOP=np.vstack((gndOP,gn))
        

    #output= np.array([x,l])
    #np.savetxt('ModelData.txt',output.T)

    
    if plotODF:
        plt.figure(dpi=150,figsize=(3.5,2.6))
        plt.plot(x,lorentzianOP[1::,:].T)
        #plt.plot(x,gndOP[1::,:].T,label='GND')
        #plt.plot(x,gaussianOP[1::,:].T,label='GD')
        plt.xlabel('Angle (\xb0)')
        plt.ylabel('Intensity (a.u.)')
        plt.xticks([0,90,180,270,360])
        plt.locator_params('y',nbins=6)
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        plt.tight_layout()
        plt.minorticks_on()
        plt.legend(('A$_2$/A$_1$=0.1','A$_2$/A$_1$=0.75'))
        plt.show() 
        plt.close()

    P2g=[]
    P2l=[]
    T2g=[]
    T2l=[]
    P2gn=[]
    T2gn=[]
    T4l=[]

    
    for i in range(1,np.shape(gaussianOP)[0]):
        cosSq3DG=np.sum(gaussianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gaussianOP[i,0:int(a/2)]*sinX[0:int(a/2)])
        P2g.append(np.round(1.5*cosSq3DG-0.5,3))
        cosSq3DL=np.sum(lorentzianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)]*sinX[0:int(a/2)])
        P2l.append(np.round(1.5*cosSq3DL-0.5,3))
        cosSq3DGN=np.sum(gndOP[i,0:int(a/2)]*cosSqX[0:int(a/2)]*sinX[0:int(a/2)])/np.sum(gndOP[i,0:int(a/2)]*sinX[0:int(a/2)])
        P2gn.append(np.round(1.5*cosSq3DGN-0.5,3))
        cosSq2DG=np.sum(gaussianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gaussianOP[i,0:int(a/2)])
        T2g.append(np.round(2*cosSq2DG-1,3))
        cosSq2DL=np.sum(lorentzianOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)])
        T2l.append(np.round(2*cosSq2DL-1,3))
        cosSq2DGN=np.sum(gndOP[i,0:int(a/2)]*cosSqX[0:int(a/2)])/np.sum(gndOP[i,0:int(a/2)])
        T2gn.append(np.round(2*cosSq2DGN-1,3))
        cosFt2DL=np.sum(lorentzianOP[i,0:int(a/2)]*cosFtX[0:int(a/2)])/np.sum(lorentzianOP[i,0:int(a/2)])
        T4l.append(np.round(8*cosFt2DL-8*cosSq2DL+1,3))
        
    
    
    fw=2*np.asarray(c)
    print('FWHM=',fw)
    print('\n <T2> vs. FWHM for A2/A1=',h)
    print('LD\n',T2l,'\nGD\n',T2g,'\nGND\n',T2gn)
    print('\n <T4> vs. FWHM for A2/A1=',h)
    print('LD\n',T4l)
    print('\n <P2> vs. FWHM for A2/A1=',h)
    print('LD\n',P2l,'\nGD\n',P2g,'\nGND\n',P2gn)


    
    #Plotting values
    
    index=0
    plt.figure(figsize=(3.5,2.6))
    plt.title('OP vs FWHM for varying A$_2$/A$_1$ (LD)')
    for i in range(0,len(T2l),len(c)):
        plt.plot(fw,T2l[i:i+len(c)],'--o', label='A$_2$/A$_1$='+str(h[index]))
        #plt.plot(fw,P2l[i:i+len(c)],'--x', label='A$_2$/A$_1$='+str(h[index]))
        index+=1
        
    plt.legend()
    plt.xlabel('FWHM (\xb0)')
    plt.ylabel('Orientation Parameter')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

    
    index=0
    plt.figure(figsize=(3.5,2.6))
    plt.title('OP vs FWHM for varying A$_2$/A$_1$ (GND)')
    for i in range(0,len(T2gn),len(c)):
        plt.plot(fw,T2gn[i:i+len(c)],'--o', label='A$_2$/A$_1$='+str(h[index]))
        #plt.plot(fw,P2gn[i:i+21],'--x', label='A$_2$/A$_1$='+str(h[index]))
        index+=1
    plt.legend()
    plt.xlabel('FWHM (\xb0)')
    plt.ylabel('Orientation Parameter')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

    index=0
    plt.figure(figsize=(3.5,2.6))
    plt.title('OP vs FWHM for varying A$_2$/A$_1$ (GD)')
    for i in range(0,len(T2g),len(c)):
        plt.plot(fw,T2g[i:i+len(c)],'--o', label='A$_2$/A$_1$='+str(h[index]))
        #plt.plot(fw,P2g[i:i+21],'--x', label='A$_2$/A$_1$='+str(h[index]))
        index+=1
    plt.legend()
    plt.xlabel('FWHM (\xb0)')
    plt.ylabel('Orientation Parameter')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

    index=0
    plt.figure(figsize=(3.5,2.6))
    plt.title('T4 vs FWHM for varying A$_2$/A$_1$ (LD)')
    for i in range(0,len(T4l),len(c)):
        plt.plot(fw,T4l[i:i+len(c)],'--o', label='A$_2$/A$_1$='+str(h[index]))
        index+=1
    plt.plot(fw,T2l[0:len(c)],'-.s', label='<T$_2$>, A$_2$/A$_1$='+str(h[0]))
    plt.legend()
    plt.xlabel('FWHM (\xb0)')
    plt.ylabel('Orientation Parameter')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

    
    '''
    # Plotting orientation parameter as function of relative areas of secondary peaks
    # activate when value of FWHM is fixed to a single value
    
    plt.figure(figsize=(3.5,2.6))
    for i in range(0,len(T2l),len(c)):
        line1, = plt.plot(h,T2l,'b-o') 
        line2, = plt.plot(h,T2gn,'g--o')
        line3, = plt.plot(h,T2g,'r-.o')
        line4, = plt.plot(h,T4l,'k:o')
        
        
    plt.legend((line1,line2,line3,line4),('<T$_2$>$_{LD}$','<T$_2$>$_{GND}$','<T$_2$>$_{GD}$','<T$_4$>$_{LD}$'))
    plt.xlabel('Relative area, A$_2$/A$_1$')
    plt.ylabel('Orientation Parameter')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
     '''

