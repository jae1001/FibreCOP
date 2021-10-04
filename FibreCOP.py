'''This program calculates the Chebyshev/Herman orientation parameters
for aligned fibres from their SEM image/intensity distribution data.
Digital photographs of macroscopic fibres can also be analysed.

First time users of a python program might need to install additional python modules
for the code to run. Please refer to python.org or similar website for installation instructions
specific to your OS.     

Preferred formats:
Image: '.tiff', imread in openCV also supports most other filetypes such as
        .jpg, .png etc. (https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)
Data: '.csv' format with no headers and only X and Y columns.

Please download the icon.gif along with the code, for smooth execution of the program
or comment out the appropriate lines in the code.

Analysis data is stored in FibreCOP_result.csv

Code developed by Dr. A. Kaniyoor,
Macromolecular Materials Laboratory,University of Cambridge, Cambridge, UK

Reference Publication:
Quantifying Alignment in Carbon Nanotube Yarns and Similar 2D Anisotropic Systems. 
A. Kaniyoor, T.S. Gspann, J. E. Mizen, J.A. Elliott, To be submitted.

'''

import math
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as tkf
from lmfit.models import LorentzianModel, LinearModel, GaussianModel, PseudoVoigtModel, Model
import scipy.signal as sp
import scipy.special as spl
from scipy.integrate import trapz
from scipy.optimize import curve_fit
import pandas as pd
import ntpath
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family']='Arial'
rcParams['font.size']=12
rcParams['lines.antialiased']=True
rcParams['mathtext.default']='regular'




def openFile(): 
    file = tkf.askopenfilename()
    entry1.delete(0,tk.END)
    entry1.insert(0, file)
     
def destroyWindows():
    plt.close('all')

def quitProgram():
    window.destroy()

def sel():
    selection=str(choice.get())
    print(selection)
    return selection

def readImage(filename):
    print('\nFilnename:',filename)
    img=cv2.imread(filename,0)   
    (h,w)=img.shape
    return img, h, w

def readFile(filename):
    print('Filnename:',filename)
    data=pd.read_csv(filename,names=['angle0','intensity0'])
    return data

def deNoiseImage(image,deNoise):   
    #Removes noise in images
    #deNoise takes values 0 to less than 1 (not 1)
    #1-deNoise gives the fraction of fourier image data to be kept,
    #0 implies no de-noising the image,
    #any fraction less than 1, say 0.3, implies 0.7 is the keep fraction 
    #in deNoise function defintion   
    #the remaining image data is treated as background

    fftImage=np.fft.fft2(image) 
    keep_fraction = 1-deNoise 
    im_fft2 = fftImage.copy()  
    r, c = im_fft2.shape       
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    deNoisedImage=np.abs(np.fft.ifft2(im_fft2))
    deNoisedImage=deNoisedImage.astype(np.uint8)
    plt.subplot(1,2,1)
    plt.imshow(image,'gray')
    plt.title('Cropped Image')
    plt.subplot(1,2,2)
    plt.imshow(deNoisedImage,'gray')
    plt.title('De-noised Image')
    plt.ion()
    plt.show()
    plt.pause(0.001)
    return deNoisedImage

def rotateImage(image,rotate):
    if rotate == 'Yes':
        rotImage=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 'No':
        rotImage=image
    (h,w)=rotImage.shape
    return rotImage,h,w

def cropImage(image,h,w,deNoise,sF,rot):
    #sF is strip fraction, refers to height percentage of SEM info band
    #for deNoise, refer the appropriate function

    sF=1-sF/100 
    h=int(round(h*sF))
    croppedImage = image[0:h,0:w]
    croppedImage,h,w = rotateImage(croppedImage,rot)
    
    if deNoise == 0:
        return croppedImage, h, w
    else:
        deNoisedImage = deNoiseImage(croppedImage,deNoise)
        (h,w)=deNoisedImage.shape
        return deNoisedImage, h, w


def makebinary(image,binarize):
    #Make Binary image
    #(https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html)
    if binarize=="Gaussian":
        image = cv2.GaussianBlur(image,(5,5),0)
        binImage=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,1)
    elif binarize=="OTSU":
        ret,binImage=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (h,w)=binImage.shape
    return binImage


def fourierTrans(image):
    #Perform Fourier transform
    #https://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
    fftImage=np.fft.fft2(image)
    fftShiftImage=np.fft.fftshift(fftImage)
    fftMagImage=np.abs(fftShiftImage)
    #fftMagImage=np.log(1+fftMagImage)
    (h,w)=fftMagImage.shape
    return fftMagImage,h,w


def createCircularMask(image, h, w, centre=None, radius=None):
    #This function creates a circular mask on the image
    #(https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-
    #python-using-numpy-arrays) 
    if centre is None:                                                          # use the middle of the image
        centre = [int(w/2), int(h/2)]
    if radius is None:                                                          # use the smallest distance btwn center & image walls
        radius = min(centre[0], centre[1], w-centre[0], h-centre[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_centre = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)

    mask = dist_from_centre <= radius
    maskedImage = image.copy()
    maskedImage[~mask]=0
    #maskedImage = cv2.GaussianBlur(maskedImage, (51, 51), 0)
    return maskedImage


def radialSum(image, binsize, mask=None, symmetric=None, centre=None):
    #This function calculates the radial sum. It is a modified form of that available at
    #https://github.com/keflavich/image_tools/blob/master/image_tools/radialprofile.py#L125'''    
    y, x = np.indices(image.shape)                                              # Calculate the indices from the image
    
    if centre is None:
        centre = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])       
        
    if mask is None:
        # mask is only used in a flat context
        mask = np.ones(image.shape,dtype='bool').ravel()
    elif len(mask.shape) > 1:
        mask = mask.ravel()

    
    theta = np.arctan2(y - centre[1], x - centre[0])                            #angle bw lines (0,0) to (0,1) and (0,0) to (y,x) 
    theta[theta < 0] += 2*np.pi                                                 #if theta less than zero, add 2pi
    theta_deg = theta*180.0/np.pi                                               # angle from 3'o clock position clockwise
    maxangle = 360
    
   
    nbins = int(np.round(maxangle / binsize))
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    bin_centers = (bins[1:]+bins[:-1])/2.0                                      # more noise in data if we use bins
    whichbin = np.digitize(theta_deg.flat,bin_centers)                          # Return the indices of the bins to which each value in input array belongs
                                                                                # which bin contains the said angle/value
    radialSum = np.array([image.flat[mask*(whichbin==b)].sum()  for b in range(1,nbins+1)])

    return bin_centers[1:-1], radialSum[1:-1]                                   #avoiding last values to avoid large fall in intensity which is an interpolation error


# Defining GND model 
# https://mike.depalatis.net/blog/lmfit.html
class gndModel(Model):
    def __init__(self, *args, **kwargs):
        def gnd_func(x, amplitude, center, sigma, beta):
            return (amplitude*beta/(2*sigma*spl.gamma(1/beta)))*(np.exp(-(np.abs(x - center)/sigma)**beta))
        super(gndModel, self).__init__(gnd_func, *args, **kwargs)

    def guess(self, data, **kwargs):
        params = self.make_params()
        def pset(param, value):
            params["%s%s" % (self.prefix, param)].set(value=value)
        pset("amplitude", np.max(data) - np.min(data))
        pset("center", x[np.max(data)])
        pset("sigma", 1)
        pset("beta", 1.5)
        
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


def fitModel (x,y,t1,t2,t3,t4,t5,t6,n,c1,c2,c3,c4,c5,c6,chck1,chck2,chck3,bl):
    fitType1=t1
    fitType2=t2
    fitType3=t3
    fitType4=t4
    fitType5=t5
    fitType6=t6
    numPk=n
    cen=(c1,c2,c3,c4,c5,c6)
    fitstats=chck1
    eqWidth=chck2
    eqBeta=chck3
    bline=bl

       
    Lin1 = LinearModel(prefix='BackG_')
    pars = Lin1.make_params()
    pars['BackG_slope'].set(0, min=-0.001, max=0.001)
    pars['BackG_intercept'].set(min(y), min=0)
    
    if fitType1=='Lorentzian':
        pk1 = LorentzianModel(prefix='Peak1_')
    elif fitType1=='Gaussian':
        pk1 = GaussianModel(prefix='Peak1_')
    elif fitType1=='PseudoVoigt':
        pk1 = PseudoVoigtModel(prefix='Peak1_')
    elif fitType1=='GND':
        pk1 = gndModel(prefix='Peak1_')
       
    if fitType2=='Lorentzian':
        pk2 = LorentzianModel(prefix='Peak2_')
    elif fitType2=='Gaussian':
        pk2 = GaussianModel(prefix='Peak2_')
    elif fitType2=='PseudoVoigt':
        pk2 = PseudoVoigtModel(prefix='Peak2_')
    elif fitType2=='GND':
        pk2 = gndModel(prefix='Peak2_')
    
    if fitType3=='Lorentzian':
        pk3 = LorentzianModel(prefix='Peak3_')
    elif fitType3=='Gaussian':
        pk3 = GaussianModel(prefix='Peak3_')
    elif fitType3=='PseudoVoigt':
        pk3 = PseudoVoigtModel(prefix='Peak3_')
    elif fitType3=='GND':
        pk3 = gndModel(prefix='Peak3_')
        
    if fitType4=='Lorentzian':
        pk4 = LorentzianModel(prefix='Peak4_')
    elif fitType4=='Gaussian':
        pk4 = GaussianModel(prefix='Peak4_')
    elif fitType4=='PseudoVoigt':
        pk4 = PseudoVoigtModel(prefix='Peak4_')
    elif fitType4=='GND':
        pk4 = gndModel(prefix='Peak4_')

    if fitType5=='Lorentzian':
        pk5 = LorentzianModel(prefix='Peak5_')
    elif fitType5=='Gaussian':
        pk5 = GaussianModel(prefix='Peak5_')
    elif fitType5=='PseudoVoigt':
        pk5 = PseudoVoigtModel(prefix='Peak5_')
    elif fitType5=='GND':
        pk5 = gndModel(prefix='Peak5_')

    if fitType6=='Lorentzian':
        pk6 = LorentzianModel(prefix='Peak6_')
    elif fitType6=='Gaussian':
        pk6 = GaussianModel(prefix='Peak6_')
    elif fitType6=='PseudoVoigt':
        pk6 = PseudoVoigtModel(prefix='Peak6_')
    elif fitType6=='GND':
        pk6 = gndModel(prefix='Peak6_')
    
      
    pars.update(pk1.make_params())
    pars['Peak1_center'].set(cen[0], min=cen[0]-10, max=cen[0]+10)
    pars['Peak1_sigma'].set(20, min=0.01, max=50)
    pars['Peak1_amplitude'].set(1e7, min=0)
    if fitType1=='GND':
            pars['Peak1_beta'].set(1.5,min=1,max=2)
                        
    if numPk==2 or numPk==3 or numPk==4 or numPk==5 or numPk==6: 
        pars.update(pk2.make_params())
        pars['Peak2_center'].set(cen[1], min=cen[1]-10, max=cen[1]+10)
        pars['Peak2_amplitude'].set(1e7, min=0)
        if eqWidth==1:
            pars['Peak2_sigma'].set(expr='Peak1_sigma')
        elif eqWidth==0:
            pars['Peak2_sigma'].set(30, min=0.01, max=50)
        if fitType2=='GND':
            if eqBeta==1:
                pars['Peak2_beta'].set(expr='Peak1_beta') 
            elif eqBeta==0:
                pars['Peak2_beta'].set(1.5,min=1,max=2)
                
    if numPk==3 or numPk==4 or numPk==5 or numPk==6:
        pars.update(pk3.make_params())
        pars['Peak3_center'].set(cen[2], min=cen[2]-10, max=cen[2]+10)
        pars['Peak3_amplitude'].set(1e7, min=0)
        if eqWidth==1:
            pars['Peak3_sigma'].set(expr='Peak1_sigma')
        elif eqWidth==0:
            pars['Peak3_sigma'].set(30, min=0.01, max=50)
        if fitType2=='GND':
            if eqBeta==1:
                pars['Peak3_beta'].set(expr='Peak1_beta') 
            elif eqBeta==0:
                pars['Peak3_beta'].set(1.5,min=1,max=2)

    if numPk==4 or numPk==5 or numPk==6: 
        pars.update(pk4.make_params())
        pars['Peak4_center'].set(cen[3], min=cen[3]-10, max=cen[3]+10)
        pars['Peak4_sigma'].set(15, min=0.01, max=50)
        pars['Peak4_amplitude'].set(1e7, min=0)
        if fitType4=='GND':
            pars['Peak4_beta'].set(1.5,min=1,max=2) 

    if numPk==5 or numPk==6:
        pars.update(pk5.make_params())
        pars['Peak5_center'].set(cen[4], min=cen[4]-10, max=cen[4]+10)
        pars['Peak5_sigma'].set(15, min=0.01, max=50)
        pars['Peak5_amplitude'].set(1e7, min=0)
        if fitType5=='GND':
            pars['Peak5_beta'].set(1.5,min=1,max=2)
            
    if numPk==6:
        pars.update(pk6.make_params())
        pars['Peak6_center'].set(cen[5], min=cen[5]-10, max=cen[5]+10)
        pars['Peak6_sigma'].set(15, min=0.01, max=50)
        pars['Peak6_amplitude'].set(1e7, min=0)
        if fitType6=='GND':
            pars['Peak6_beta'].set(1.5,min=1,max=2)
            
    #model definition
    pkModel=Lin1
    
    if numPk==2:
        pkModel+=pk1+pk2
    elif numPk==3:
        pkModel+=pk1+pk2+pk3
    elif numPk==4:
        pkModel+=pk1+pk2+pk3+pk4
    elif numPk==5:
        pkModel+=pk1+pk2+pk3+pk4+pk5 
    elif numPk==6:
        pkModel+=pk1+pk2+pk3+pk4+pk5+pk6         
       
   
    out = pkModel.fit(y, pars, x=x, weights=1.0/y)

    if fitstats==1:
        print('\n',out.fit_report(show_correl=False))
    
            
    plt.figure(dpi=150, figsize=(3.5,2.8))
    lwid=2
    #plt.title('Radial Intensity distribution',fontsize=16)
    plt.plot(x,y, label='data',lw=2)
    plt.plot(x,out.best_fit,'r-',lw=lwid,label='fit')
    plt.xlabel('Angle (\xb0)', fontsize=16)
    plt.ylabel('Intensity (a.u.)',fontsize=16)
    plt.xticks([0,90,180,270,360],fontsize=14)
    plt.locator_params('y',nbins=6)
    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.minorticks_on()
    #plt.legend(fontsize=10)

    plot_components =True
    if plot_components:
        comps = out.eval_components(x=x)
        plt.plot(x, comps['Peak1_']+comps['BackG_'], 'c--',lw=lwid)
        plt.plot(x, comps['Peak2_']+comps['BackG_'], 'b--',lw=lwid)
        if numPk==3:
            plt.plot(x, comps['Peak3_']+comps['BackG_'], 'y--',lw=lwid)
        if numPk==4:
            plt.plot(x, comps['Peak3_']+comps['BackG_'], 'y--',lw=lwid)
            plt.plot(x, comps['Peak4_']+comps['BackG_'], 'g--',lw=lwid)
        if numPk==5:
            plt.plot(x, comps['Peak3_']+comps['BackG_'], 'y--',lw=lwid)
            plt.plot(x, comps['Peak4_']+comps['BackG_'], 'g--',lw=lwid)   
            plt.plot(x, comps['Peak5_']+comps['BackG_'], 'm--',lw=lwid)
        if numPk==6:
            plt.plot(x, comps['Peak3_']+comps['BackG_'], 'y--',lw=lwid)
            plt.plot(x, comps['Peak4_']+comps['BackG_'], 'g--',lw=lwid)   
            plt.plot(x, comps['Peak5_']+comps['BackG_'], 'm--',lw=lwid)
            plt.plot(x, comps['Peak6_']+comps['BackG_'], 'k--',lw=lwid)
        plt.plot(x, comps['BackG_'], 'r--',lw=lwid)
        #plt.title('Radial Intensity Distribution', fontsize=14)
        #plt.xlabel('Angle (\xb0)', fontsize=28)
        #plt.ylabel('Intensity (a.u.)', fontsize=28)
    #plt.show()

    if bline=="Auto":
        yBestFit = out.best_fit - comps['BackG_']
    if bline=="None":
        yBestFit = out.best_fit
    if bline=="Constant":
        yBestFit = out.best_fit - 0.5*comps['BackG_'] # arbitrary, can be changed later
            
    return out, yBestFit

            
    
# calculating cos squred and cos fourth for the two-dimensional ODF    
def CosNth2D (x,y,Range='Pi'):
    if Range == '2Pi':
        div=1
    elif Range == 'Pi':
        div=2
    elif Range == 'Pi/2':
        div=4
    x=x[0:int(len(x)/div)]
    yAvg=(y[0:math.floor(len(y)/div)]+y[math.ceil(len(y)/div)::])/2  # taking average of values upto Pi
    num=yAvg*np.power(np.cos(x*np.pi/180),2)
    den=yAvg
    cosSqrd2D=round(trapz(num,x)/trapz(den,x),3)
    num1=yAvg*np.power(np.cos(x*np.pi/180),4)
    cosFourth2D=round(trapz(num1,x)/trapz(den,x),3)
    return cosSqrd2D, cosFourth2D

# calculating cos squred and cos fourth for the three-dimensional ODF 
def CosNth3D (x,y,Range='Pi'):
    #If range is 2Pi, then cos sqaured goes to negative due to sin term
    if Range == '2Pi':
        div=1
    elif Range == 'Pi':
        div=2
    elif Range == 'Pi/2':
        div=4
    x=x[0:int(len(x)/div)]
    y=(y[0:math.floor(len(y)/div)]+y[math.ceil(len(y)/div)::])/2  # taking average of values upto Pi
    num=y*np.power(np.cos(x*np.pi/180),2)*np.abs(np.sin(x*np.pi/180))
    den=y*np.abs(np.sin(x*np.pi/180))
    cosSqrd3D=round(trapz(num,x)/trapz(den,x),3)
    num1=y*np.power(np.cos(x*np.pi/180),4)*np.abs(np.sin(x*np.pi/180))
    cosFourth3D=round(trapz(num1,x)/trapz(den,x),3)
    return cosSqrd3D, cosFourth3D


def ChebT2 (x):
    T2 = round((2*x-1),3)
    return T2

def ChebT4 (x,y):
    T4=round((8*y-8*x+1),3)
    return T4      

def HermanP2 (x):  
    P2 = round((1.5*x-0.5),3)
    return P2

def HermanP4 (x,y):
    P4 = round((4.375*y-3.75*x+0.375),3)
    return P4  



    
''' Main code of the program. Calls for the above functions'''

def calcCOP():
    filename=entry1.get() #getting input parameters
    selection=choice.get()
    stripF=float(stripHeight.get())
    numScan=int(xScan.get())-1
    binSize=float(BinSize.get())
    dNoise=float(Denoise.get())
    binarize=Binarize.get()
    disp=DispImg.get()
    filtLevel=int(filtLev.get())
    rotate=RotateImg.get()
    t1=fitTyp1.get()
    t2=fitTyp2.get()
    t3=fitTyp3.get()
    t4=fitTyp4.get()
    t5=fitTyp5.get()
    t6=fitTyp6.get()
    n=int(noPk.get())
    c1=int(cen1.get())
    c2=int(cen2.get())
    c3=int(cen3.get())
    c4=int(cen4.get())
    c5=int(cen5.get())
    c6=int(cen6.get())
    check1=checkvar1.get()
    check2=checkvar2.get()
    check3=checkvar3.get()
    bl=bltype.get()
    
    filebasename=ntpath.basename(filename)
    
    
    '''Image Analysis'''
    if selection==1: # for analysing images 
        origImage,h1,w1 = readImage(filename) #Aquire Image
        print('Original image size is', h1, 'x', w1)
        croppedImage,h2,w2 = cropImage(origImage,h1,w1,deNoise=dNoise,sF=stripF,rot=rotate)
        print('Cropped image size is', h2, 'x', w2,'\n')

        #To determine a square area with sides equal to an integral multiple of 256
        if h2<=w2:
            maxSqr=256*(h2//256)
        else:
            maxSqr=256*(w2//256)
        col=maxSqr
        row=col

        #To scan the image
        
        if h2<=w2:
            diff=w2-col
            if numScan > 0:
                stepsize = diff//numScan
            elif numScan == 0:
                stepsize = diff+1
        else:
            diff=h2-row
            if numScan > 0:
                stepsize = diff//numScan
            elif numScan == 0:
                stepsize = diff+1
        
        # To get x and y data by scanning the image
        angle=[]
        intensity=[]
        
        plt.figure()
        for i in range(0,diff+1,stepsize):
            if h2<=w2: processedImage=croppedImage[0:row,i:col+i]
            else: processedImage=croppedImage[i:row+i,0:col] 
            print('Processed Image size is', np.shape(processedImage))
            plt.imshow(processedImage,'gray')
            plt.title('Processed Image')
            plt.ion()
            plt.show()
            plt.pause(.001)
            

            '''Fourier Transform'''
            binImage = makebinary(processedImage,binarize)
            fourierImage,h,w = fourierTrans(binImage)
            maskedImage = createCircularMask(fourierImage, h, w) #Draw a circular profile by creating a mask i.e making points outside circle, zero                                          

            '''Calling the main radialSum fucntion'''
            angle0, intensity0 = radialSum(maskedImage, binsize=binSize)
            
            if i==0:
               angle=np.append(angle,angle0)
               intensity=np.append(intensity,intensity0)
            else:
               angle=np.vstack((angle,angle0))
               intensity=np.vstack((intensity,intensity0))

        

        #Image Plotting
        plt.close()

        plotFigures = disp
        if plotFigures=='Yes':
            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(processedImage,'gray')
            plt.title('Processed Image')
            plt.axis('off')
            
            #plt.figure()
            plt.subplot(2,2,2)
            plt.imshow(binImage,'gray')
            plt.title('Binary Image')
            plt.axis('off')

            #plt.figure()
            plt.subplot(2,2,3)
            plt.imshow(fourierImage, norm=LogNorm())
            plt.title('Fourier Image')
            plt.axis('off')

            #plt.figure()
            plt.subplot(2,2,4)
            plotImage=np.log(1+fourierImage)
            plotImage = createCircularMask(plotImage, h, w)
            plt.imshow(plotImage,'gray')
            #plt.imshow(maskedImage,'gray')
            plt.title('Masked Image')
            plt.axis('off')
            plt.ion()
            plt.show()
            
            plt.pause(0.001)


        angle=angle.T
        intensity=intensity.T
        
    elif selection==2: # for analysing given data
        data=readFile(filename)
        angle=data.angle0[1:].to_numpy()
        intensity=data.intensity0[1:].to_numpy()
        if rotate =='Yes':
            angle=angle-90
    
    
    plotRawdata= disp
    if plotRawdata=='Yes':
        plt.figure()
        plt.plot(angle,intensity)
        plt.title('Radial Intensity distribution (raw data)', fontsize=16)
        plt.ylabel('Intensity (a.u.)', fontsize=18)
        plt.xlabel('Angle (\xb0)',fontsize=18)
        #plt.legend()
        plt.tight_layout()
        plt.ion()
        plt.show()

        
    #Calculation
    
    fitOutput=[]
    IDFtype=[]
    binSiz=[]
    numofPeak=[]
    CosSq2Dfilt=[]
    CosQt2Dfilt=[]
    T2filt=[]
    T4filt=[]
    CosSq3Dfilt=[]
    CosQt3Dfilt=[]
    P2filt=[]
    P4filt=[]
    
    #Note: angle is stack of columns, number of cloumns=number of scans  
    try:
        count=np.shape(angle)[1]
    except:
        count=1
     
    for i in range(0,count):
        if count==1:
           x=angle
           filteredIntensity = sp.medfilt(intensity, filtLevel)
        else:
           x=angle[:,i]
           filteredIntensity = sp.medfilt(intensity[:,i], filtLevel)

                
        plt.figure()
        plt.plot(x,filteredIntensity)
        plt.title('Radial Intensity distribution (filtered)', fontsize=16)
        plt.ylabel('Intensity (a.u.)', fontsize=18)
        plt.xlabel('Angle (\xb0)',fontsize=18)
            
                       
        #fitting peaks to filteredintensity        
        fitOut,yFilter=fitModel(x,filteredIntensity,t1,t2,t3,t4,t5,t6,n,c1,c2,c3,c4,c5,c6,check1,check2,check3,bl)
        fitOutput.append(fitOut)
        IDFtype.append(t2)
        binSiz.append(binSize)
        numofPeak.append(n)
        
        #choosing the right principal/reference axis
        maxAngF=x[yFilter[0:len(yFilter)//2].argmax()]
        if maxAngF>45 and maxAngF<=135:
            shift=90-maxAngF
        elif maxAngF>135 and maxAngF<=215:
            shift=180-maxAngF
        elif maxAngF<=45 and maxAngF>=0:
            shift=0-maxAngF
        xNew=x+shift
        print('Peak of fit curve is at',maxAngF,'degrees')
        print('Principal axis for filtered peak shifted by',shift,'degrees')         

        plotShiftedODF='No'
        if plotShiftedODF=='Yes':
            plt.figure()
            plt.plot(xNew,filteredIntensity)
            plt.title('Shifted Radial Intensity distribution', fontsize=16)
            plt.ylabel(r'Intensity (a.u.)', fontsize=18)
            plt.xlabel(r'Orientation (degrees)',fontsize=18)
            #plt.legend()
            plt.show()

        #cos squred calculation for all scans
        CosSq2Dfilt.append(CosNth2D(xNew,yFilter)[0])
        CosQt2Dfilt.append(CosNth2D(xNew,yFilter)[1])
        CosSq3Dfilt.append(CosNth3D(xNew,yFilter)[0])
        CosQt3Dfilt.append(CosNth3D(xNew,yFilter)[1])
        
   
        
    # Calculating Orientation parameter for all scans
    for i in range(0,len(CosSq2Dfilt)):
        T2filt.append(ChebT2(CosSq2Dfilt[i]))
    for i in range(0,len(CosQt2Dfilt)):
        T4filt.append(ChebT4(CosSq2Dfilt[i],CosQt2Dfilt[i]))
    for i in range(0,len(CosSq3Dfilt)):
        P2filt.append(HermanP2(CosSq3Dfilt[i]))
    for i in range(0,len(CosQt3Dfilt)):
        P4filt.append(HermanP4(CosSq3Dfilt[i],CosQt3Dfilt[i]))
        
 
    #calculating average
        
    #CosSq2DfiltMean=np.round(np.mean(CosSq2Dfilt),3)
    #CosSq2DfiltDev=np.round(np.std(CosSq2Dfilt),3)
    #CosQt2DfiltMean=np.round(np.mean(CosQt2Dfilt),3)
    #CosQt2DfiltDev=np.round(np.std(CosQt2Dfilt),3)
    T2filtAvg=np.round(np.mean(T2filt),3)
    T2filtDev=np.round(np.std(T2filt),3)
    T4filtAvg=np.round(np.mean(T4filt),3)
    T4filtDev=np.round(np.std(T4filt),3)

    #CosSq3DfiltMean=np.round(np.mean(CosSq3Dfilt),3)
    #CosSq3DfiltDev=np.round(np.std(CosSq3Dfilt),3)
    #CosQt3DfiltMean=np.round(np.mean(CosQt3Dfilt),3)
    #CosQt3DfiltDev=np.round(np.std(CosQt3Dfilt),3)
    P2filtAvg=np.round(np.mean(P2filt),3)
    P2filtDev=np.round(np.std(P2filt),3)
    P4filtAvg=np.round(np.mean(P4filt),3)
    P4filtDev=np.round(np.std(P4filt),3)
    
    #calculating FWHM
    FWHM=[]
    peak2beta=[]
    name=[]
    AIC=[]
    BIC=[]
    RedChi2=[]
    FitVar=[]
    for i in range(0,len(fitOutput)):        
        peak2sigma=np.round(fitOutput[i].params['Peak2_sigma'].value,2)
        redchi=fitOutput[i].redchi
        aic=fitOutput[i].aic
        bic=fitOutput[i].bic
        numfitvar=fitOutput[i].nvarys
        
        if t2=="Lorentzian" or t2=="PseudoVoigt":
            fullwidth=np.round(2.0*peak2sigma,2)
            peak2beta=np.full_like(fitOutput,np.nan,dtype=float)
        elif t2=="Gaussian":
            fullwidth=np.round(2.3548*peak2sigma,2)
            peak2beta=np.full_like(fitOutput,np.nan,dtype=float)
        elif t2=="GND":
            pk2beta=np.round(fitOutput[i].params['Peak2_beta'].value,2)
            peak2beta.append(pk2beta)
            fullwidth=np.round(2*1.414*peak2sigma*((np.log(2))**(1/pk2beta)),2)
        FWHM.append(fullwidth)
        name.append(filebasename)
        AIC.append(aic)
        BIC.append(bic)
        RedChi2.append(redchi)
        FitVar.append(numfitvar)

    FWHMavg=np.round(np.mean(FWHM),2)
    FWHMdev=np.round(np.std(FWHM),2)
    peak2betaAvg=np.round(np.mean(peak2beta),2)
    peak2betadev=np.round(np.std(peak2beta),2)
    
    print ('\nThe Orientation Parameters for the sample are')
    #print('\t cos^2Theta =', CosSq2Dfilt,', Average cos^2Theta =', CosSq2DfiltMean,'+/-',CosSq2DfiltDev)
    #print('\t cos^4Theta =', CosQt2Dfilt,', Average cos^4Theta =', CosQt2DfiltMean,'+/-',CosQt2DfiltDev)
    #print('\t cos^2Theta =', CosSq3Dfilt,', Average cos^2Theta =', CosSq3DfiltMean,'+/-',CosSq3DfiltDev)
    #print('\t cos^3Theta =', CosQt3Dfilt,', Average cos^4Theta =', CosQt3DfiltMean,'+/-',CosQt3DfiltDev)
    print('\t Fitted IDF =',t2)
    print('\t Number of fitted peaks =', numofPeak)
    print('\t FWHM (Peak 2) =',FWHM,', Average FWHM =',FWHMavg,'+/-',FWHMdev)
    if t2=="GND":
        print('\t Beta(shape factor)=',peak2beta,', Average Beta =',peak2betaAvg,'+/-',peak2betadev)
    print('\t Chebyshev T2 =',T2filt,', Average T2 =', T2filtAvg,'+/-',T2filtDev)
    print('\t Chebyshev T4 =',T4filt,', Average T4 =', T4filtAvg,'+/-',T4filtDev)
    print('\t Hermann P2 =',P2filt,', Average P2 =', P2filtAvg,'+/-',P2filtDev)
    print('\t Hermann P4 =',P4filt,', Average P4 =', P4filtAvg,'+/-',P4filtDev)
    
    print('\nThank you for using FibreCOP.')
    print('Your data is saved in the file, FibreCOP_result.csv.')        

    #saving the data in a .csv file        
    output=np.asarray([name,binSiz,numofPeak,IDFtype,FWHM,peak2beta,T2filt,T4filt,P2filt,P4filt,FitVar,RedChi2,AIC,BIC])
    output=output.T
    C=('Sample','Bin Size','Peaks Fit','IDF','FWHM','Beta(GND)','T2','T4','P2','P4','Parameters Fit','Reduced Chi Sq','AIC','BIC')
    pd.DataFrame(output).to_csv("FibreCOP_result.csv", mode='a',index=False,header=C)





#GUI
    
window = tk.Tk() #Create window object
window.title('FibreCOP: Chebyshev Orientation Parameter for CNT textiles')
#window.geometry("600x780+0+0")

can_icon = tk.Canvas(window,height=50,width=60,relief='sunken')
can_icon.grid(row=0,rowspan=5,column=0,columnspan=1, sticky=tk.NSEW)
imfile = tk.PhotoImage(file = "icon.gif")
imfile= imfile.subsample(3,3)
image = can_icon.create_image(30,30, anchor=tk.CENTER, image=imfile)

label1 = tk.Label(window, text="Enter File Path")
label1.grid(row=0,column=1, pady=5)
filePath=tk.StringVar()
entry1=tk.Entry(window,width=40,textvariable=filePath)
entry1.grid(row=0,column=2,columnspan=3,sticky=tk.EW, padx=5, pady=5)
browseButton = tk.Button(window, text='Browse', command=openFile)
browseButton.grid(row=0,column=5,sticky=tk.W, pady=5)

label56 = tk.Label(window, text="Data Type") 
label56.grid(row=1,column=2,sticky=tk.W)
choice=tk.IntVar()
radio1=tk.Radiobutton(window,text="Image",variable=choice,value=1)
radio1.grid(row=1,column=3,sticky=tk.W)
radio2=tk.Radiobutton(window,text="Data",variable=choice,value=2)
radio2.grid(row=1,column=4,sticky=tk.E)

label456 = tk.Label(window, text="") 
label456.grid(row=5,column=1)

can2 = tk.Canvas(window, height=300, width=30,relief='sunken')
can2.grid(row=6,rowspan=10,column=0)
can2.create_text(10, 150, text = "Image Analysis Options", angle = 90,justify="center",font=('Helevetica',13), fill="Maroon")

label4 = tk.Label(window, text="Strip height") 
label4.grid(row=6,column=1, sticky=tk.W,padx=5, pady=5)
stripHeight=tk.StringVar()
stripHeight.set("7")
entry4=tk.Entry(window,width=12, textvariable=stripHeight)
entry4.grid(row=6,column=2, padx=5, pady=5)
label14=tk.Label(window, text="(Image: %height of SEM info bar to be stripped; Data: NA)")
label14.grid(row=6,column=3, columnspan=4,sticky=tk.W,padx=5, pady=5)


label5 = tk.Label(window, text="De-noising") 
label5.grid(row=7,column=1, sticky=tk.W,padx=5, pady=5)
Denoise=tk.StringVar()
Denoise.set("0")
entry5=tk.Entry(window,width=12,textvariable=Denoise)
entry5.grid(row=7,column=2,  padx=5,pady=5)
label15=tk.Label(window, text="(Image: noise level 0-1, Data: use 0)")
label15.grid(row=7,column=3, columnspan=4,sticky=tk.W,padx=5, pady=5)

labe57 = tk.Label(window, text="Rotate") 
labe57.grid(row=8,column=1, sticky=tk.W,padx=5, pady=5)
RotateImg=tk.StringVar()
RotateImg.set("Yes")
option57=tk.OptionMenu(window,RotateImg,"Yes","No")
option57.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option57.grid(row=8,column=2, padx=5,pady=5)
label57=tk.Label(window, text="(Image: 'Yes' if image horizontal, Data: usually 'No')")
label57.grid(row=8,column=3,columnspan=4, sticky=tk.W,padx=5, pady=5)


label58 = tk.Label(window, text="Binarization") 
label58.grid(row=9,column=1, sticky=tk.W,padx=5, pady=5)
Binarize=tk.StringVar()
Binarize.set("Gaussian")
option58=tk.OptionMenu(window,Binarize,"Gaussian","OTSU")
option58.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option58.grid(row=9,column=2, padx=5,pady=5)
label158=tk.Label(window, text="(works for images only)")
label158.grid(row=9,column=3, columnspan=4,sticky=tk.W,padx=5, pady=5)

label6 = tk.Label(window, text="No. of Scans") 
label6.grid(row=10,column=1, sticky=tk.W,padx=5, pady=5)
xScan=tk.StringVar()
xScan.set("1")
entry6=tk.Entry(window,width=12,textvariable=xScan)
entry6.grid(row=10,column=2,pady=5)
label16=tk.Label(window, text="(Image: no. of sq. areas to be scanned, > 0; Data: use 1)")
label16.grid(row=10,column=3,columnspan=4, sticky=tk.W,padx=5, pady=5)

label7 = tk.Label(window, text="Bin Size") 
label7.grid(row=11,column=1, sticky=tk.W,padx=5, pady=5)
BinSize=tk.StringVar()
BinSize.set("0.25")
entry7=tk.Entry(window,width=12,textvariable=BinSize)
entry7.grid(row=11,column=2,pady=5)
label17=tk.Label(window, text="(Image: use < 1, angle step-size for radial sum; Data: NA)")
label17.grid(row=11,column=3,columnspan=4, sticky=tk.W,padx=5, pady=5)

label45 = tk.Label(window, text="Display Images") 
label45.grid(row=12,column=1, sticky=tk.W,padx=5, pady=5)
DispImg=tk.StringVar()
DispImg.set("Yes")
option45=tk.OptionMenu(window,DispImg,"Yes","No")
option45.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option45.grid(row=12,column=2, padx=5, pady=5)
label145=tk.Label(window, text="(display images used for analysis)")
label145.grid(row=12,column=3,columnspan=4, sticky=tk.W,padx=5, pady=5)

labe20 = tk.Label(window, text="Filter Interval") 
labe20.grid(row=14,column=1, sticky=tk.W,padx=5, pady=5)
filtLev=tk.StringVar()
filtLev.set("5")
entry20=tk.Entry(window,width=12,textvariable=filtLev)
entry20.grid(row=14,column=2,pady=5)
label20=tk.Label(window, text="(>=3, odd, window size for median filter)")
label20.grid(row=14,column=3,columnspan=4, sticky=tk.W,padx=5, pady=5)


labe47 = tk.Label(window, text="") 
labe47.grid(row=16,column=1)


can3 = tk.Canvas(window,  height=300, width=30,relief='sunken')
can3.grid(row=17,rowspan=10,column=0)
can3.create_text(10, 150, text = "Peak Fitting Options", angle = 90,justify="center",font=('Helevetica',13), fill="Maroon")



label29 = tk.Label(window, text="No. of Peaks") 
label29.grid(row=18,column=1, sticky=tk.W,padx=5, pady=5)
noPk=tk.StringVar()
noPk.set("3")
option29=tk.OptionMenu(window,noPk,"2","3","4","5","6")
option29.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option29.grid(row=18,column=2,pady=5)
label49=tk.Label(window, text="(min. 3/2 for horizontal/vertical orientation)")
label49.grid(row=18,column=3,columnspan=4, sticky=tk.W,padx=5, pady=5)


label22 = tk.Label(window, text="Peak 1") 
label22.grid(row=19,column=1, sticky=tk.W,padx=5, pady=5)
fitTyp1=tk.StringVar()
fitTyp1.set("Lorentzian")
option22=tk.OptionMenu(window,fitTyp1,"Lorentzian","Gaussian","PseudoVoigt","GND")
option22.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option22.grid(row=19,column=2,pady=5, padx=5)


label23 = tk.Label(window, text="Peak 2") 
label23.grid(row=20,column=1, sticky=tk.W,padx=5, pady=5)
fitTyp2=tk.StringVar()
fitTyp2.set("Lorentzian")
option23=tk.OptionMenu(window,fitTyp2,"Lorentzian","Gaussian","PseudoVoigt","GND")
option23.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option23.grid(row=20,column=2,pady=5,padx=5)

label24 = tk.Label(window, text="Peak 3") 
label24.grid(row=21,column=1, sticky=tk.W,padx=5, pady=5)
fitTyp3=tk.StringVar()
fitTyp3.set("Lorentzian")
option24=tk.OptionMenu(window,fitTyp3,"Lorentzian","Gaussian","PseudoVoigt","GND")
option24.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option24.grid(row=21,column=2,pady=5,padx=5)


label25 = tk.Label(window, text="Peak 4") 
label25.grid(row=22,column=1, sticky=tk.W,padx=5, pady=5)
fitTyp4=tk.StringVar()
fitTyp4.set("Lorentzian")
option25=tk.OptionMenu(window,fitTyp4,"Lorentzian","Gaussian","PseudoVoigt","GND")
option25.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option25.grid(row=22,column=2,pady=5,padx=5)

label26 = tk.Label(window, text="Peak 5") 
label26.grid(row=23,column=1, sticky=tk.W,padx=5, pady=5)
fitTyp5=tk.StringVar()
fitTyp5.set("Lorentzian")
option26=tk.OptionMenu(window,fitTyp5,"Lorentzian","Gaussian","PseudoVoigt","GND")
option26.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option26.grid(row=23,column=2,pady=5,padx=5)


label27 = tk.Label(window, text="Peak 6") 
label27.grid(row=24,column=1, sticky=tk.W,padx=5, pady=5)
fitTyp6=tk.StringVar()
fitTyp6.set("Lorentzian")
option27=tk.OptionMenu(window,fitTyp6,"Lorentzian","Gaussian","PseudoVoigt","GND")
option27.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option27.grid(row=24,column=2,pady=5, padx=5)

label32 = tk.Label(window, text="Centre") 
label32.grid(row=19,column=3,sticky=tk.EW, pady=5)
cen1=tk.StringVar()
cen1.set("1")
entry32=tk.Entry(window,width=10,textvariable=cen1)
entry32.grid(row=19,column=4,pady=5,sticky=tk.W)

label33 = tk.Label(window, text="Centre") 
label33.grid(row=20,column=3,sticky=tk.EW, pady=5)
cen2=tk.StringVar()
cen2.set("180")
entry33=tk.Entry(window,width=10,textvariable=cen2)
entry33.grid(row=20,column=4,pady=5,sticky=tk.W)

label34 = tk.Label(window, text="Centre") 
label34.grid(row=21,column=3,sticky=tk.EW, pady=5)
cen3=tk.StringVar()
cen3.set("359")
entry34=tk.Entry(window,width=10,textvariable=cen3)
entry34.grid(row=21,column=4,pady=5,sticky=tk.W)

label35 = tk.Label(window, text="Centre") 
label35.grid(row=22,column=3,sticky=tk.EW, pady=5)
cen4=tk.StringVar()
cen4.set("180")
entry35=tk.Entry(window,width=10,textvariable=cen4)
entry35.grid(row=22,column=4,pady=5,sticky=tk.W)

label36 = tk.Label(window, text="Centre") 
label36.grid(row=23,column=3,sticky=tk.EW, pady=5)
cen5=tk.StringVar()
cen5.set("1")
entry36=tk.Entry(window,width=10,textvariable=cen5)
entry36.grid(row=23,column=4,pady=5,sticky=tk.W)

label37= tk.Label(window, text="Centre") 
label37.grid(row=24,column=3,sticky=tk.EW, pady=5)
cen6=tk.StringVar()
cen6.set("359")
entry37=tk.Entry(window,width=10,textvariable=cen6)
entry37.grid(row=24,column=4,pady=5,sticky=tk.W)

checkvar1=tk.IntVar()
checkvar1.set(0)
check1=tk.Checkbutton(window,text='Fit Statistics',variable=checkvar1,onvalue=1,offvalue=0)
check1.grid(row=22,column=5,padx=5, sticky=tk.NW)

checkvar2=tk.IntVar()
checkvar2.set(1)
check2=tk.Checkbutton(window,text='Equal Peak widths',variable=checkvar2,onvalue=1,offvalue=0)
check2.grid(row=19,column=5,padx=5, pady=5,sticky=tk.W)

checkvar3=tk.IntVar()
checkvar3.set(1)
check3=tk.Checkbutton(window,text='Equal beta for GND',variable=checkvar3,onvalue=1,offvalue=0)
check3.grid(row=20,column=5,padx=5, pady=5,sticky=tk.W)

label124=tk.Label(window,text='(options valid up to 3 peaks)')
label124.grid(row=21,column=5,sticky=tk.N)

label156=tk.Label(window,text='Baseline Removal')
label156.grid(row=23,column=5, sticky=tk.W)
bltype=tk.StringVar()
bltype.set("Auto")
option156=tk.OptionMenu(window,bltype,"Auto","None","Constant")
option156.configure(width=10,bg='white',bd=1,activebackground='white',relief='sunken')
option156.grid(row=24,column=5,sticky=tk.NW,padx=5)

calcButton = tk.Button(window, text='Calculate COP', command=calcCOP)
calcButton.grid(row=100,column=2, columnspan = 3, sticky= tk.EW, padx=5, pady=5)

closeButton = tk.Button(window, text='Close Graphs', command=destroyWindows)
closeButton.grid(row=100,column=0, columnspan=2, sticky= tk.EW, padx=5, pady=5)

quitButton = tk.Button(window, text="Quit",command=quitProgram) 
quitButton.grid(row=100,column=5,columnspan=2,sticky=tk.EW,padx=5, pady=5)

label72= tk.Label(window, text="@ AK2011, Macromolecular Materials Laboratory, University of Cambridge, 2020", font=('Helevetica',7)) 
label72.grid(row=101,column=1,columnspan=5,sticky=tk.EW, padx=5)

window.mainloop()



''' #Program Ends Here'''

