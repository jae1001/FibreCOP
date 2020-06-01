
'''This program calculates the Chebyshev/Herman orientation parameters
for aligned fibres from their SEM image.
Authored by Dr. Adarsh Kaniyoor,
Macromolecular Materials Laboratory,University of Cambridge, Cambridge, UK
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tkinter as tk 
import tkinter.filedialog as tkf
from lmfit.models import LorentzianModel, LinearModel, GaussianModel, PseudoVoigtModel
import scipy.signal as sp
from scipy.integrate import trapz
import pandas as pd

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
    print('Filnename:',filename)
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
    #any fraction less than 1, say 0.3, implies 0.7 is keep fraction 
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


def makebinary(image):
    #Make Binary image
    #(https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html)
    image = cv2.GaussianBlur(image,(5,5),0)
    binImage=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,1)
    #ret,binImage=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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



def fitModel (x,y,t1,t2,t3,t4,t5,t6,n,c1,c2,c3,c4,c5,c6):
    fitType1=t1
    fitType2=t1
    fitType3=t3
    fitType4=t4
    fitType5=t5
    fitType6=t6
    numPk=n
    cen=(c1,c2,c3,c4,c5,c6)
    
    Lin1 = LinearModel(prefix='BackG_')
    pars = Lin1.make_params()
    pars['BackG_slope'].set(0, min=-0.001, max=0.001)
    pars['BackG_intercept'].set(2e7, min=0)
    
    if fitType1=='Lorentzian':
        pk1 = LorentzianModel(prefix='Peak1_')
    elif fitType1=='Gaussian':
        pk1 = GaussianModel(prefix='Peak1_')
    elif fitType1=='PseudoVoigt':
        pk1 = PseudoVoigtModel(prefix='Peak1_')

    if fitType2=='Lorentzian':
        pk2 = LorentzianModel(prefix='Peak2_')
    elif fitType2=='Gaussian':
        pk2 = GaussianModel(prefix='Peak2_')
    elif fitType2=='PseudoVoigt':
        pk2 = PseudoVoigtModel(prefix='Peak2_')

    if fitType3=='Lorentzian':
        pk3 = LorentzianModel(prefix='Peak3_')
    elif fitType3=='Gaussian':
        pk3 = GaussianModel(prefix='Peak3_')
    elif fitType3=='PseudoVoigt':
        pk3 = PseudoVoigtModel(prefix='Peak3_')

    if fitType4=='Lorentzian':
        pk4 = LorentzianModel(prefix='Peak4_')
    elif fitType4=='Gaussian':
        pk4 = GaussianModel(prefix='Peak4_')
    elif fitType4=='PseudoVoigt':
        pk4 = PseudoVoigtModel(prefix='Peak4_')

    if fitType5=='Lorentzian':
        pk5 = LorentzianModel(prefix='Peak5_')
    elif fitType5=='Gaussian':
        pk5 = GaussianModel(prefix='Peak5_')
    elif fitType5=='PseudoVoigt':
        pk5 = PseudoVoigtModel(prefix='Peak5_')

    if fitType6=='Lorentzian':
        pk6 = LorentzianModel(prefix='Peak6_')
    elif fitType6=='Gaussian':
        pk6 = GaussianModel(prefix='Peak6_')
    elif fitType6=='PseudoVoigt':
        pk6 = PseudoVoigtModel(prefix='Peak6_')
    
      
    pars.update(pk1.make_params())
    pars['Peak1_center'].set(cen[0], min=cen[0]-10, max=cen[0]+10)
    pars['Peak1_sigma'].set(20, min=0.1, max=50)
    pars['Peak1_amplitude'].set(1e7, min=0)
            
    pars.update(pk2.make_params())
    pars['Peak2_center'].set(cen[1], min=cen[1]-10, max=cen[1]+10)
    pars['Peak2_sigma'].set(30, min=0.1, max=50)
    pars['Peak2_amplitude'].set(1e7, min=0)
            
    pars.update(pk3.make_params())
    pars['Peak3_center'].set(cen[2], min=cen[2]-10, max=cen[2]+10)
    pars['Peak3_sigma'].set(20, min=0.1, max=50)
    pars['Peak3_amplitude'].set(1e7, min=0)

    pars.update(pk4.make_params())
    pars['Peak4_center'].set(cen[3], min=cen[3]-10, max=cen[3]+10)
    pars['Peak4_sigma'].set(15, min=0.1, max=50)
    pars['Peak4_amplitude'].set(1e7, min=0)

    pars.update(pk5.make_params())
    pars['Peak5_center'].set(cen[4], min=cen[4]-10, max=cen[4]+10)
    pars['Peak5_sigma'].set(15, min=0.1, max=50)
    pars['Peak5_amplitude'].set(1e7, min=0)

    pars.update(pk6.make_params())
    pars['Peak6_center'].set(cen[5], min=cen[5]-10, max=cen[5]+10)
    pars['Peak6_sigma'].set(15, min=0.1, max=50)
    pars['Peak6_amplitude'].set(1e7, min=0)
    
    
    if numPk==2:
        pkMod=pk1+pk2
    elif numPk==3:
        pkMod=pk1+pk2+pk3
    elif numPk==4:
        pkMod=pk1+pk2+pk3+pk4
    elif numPk==5:
        pkMod=pk1+pk2+pk3+pk4+pk5 
    elif numPk==6:
        pkMod=pk1+pk2+pk3+pk4+pk5+pk6         
       
   
    pkModel = Lin1+pkMod # model definition
    out = pkModel.fit(y, pars, x=x)
    print(out.fit_report(min_correl=0.25))
        
    plt.figure()
    plt.title('Radial Intensity distribution', fontsize=14)
    plt.plot(x,y, label='filtered')
    plt.plot(x,out.best_fit,'r-',lw=2,label='fit')
    plt.legend()

    plot_components = True
    if plot_components:
        comps = out.eval_components(x=x)
        plt.plot(x, comps['Peak1_']+comps['BackG_'], 'c--')
        plt.plot(x, comps['Peak2_']+comps['BackG_'], 'b--')
        if numPk==3:
            plt.plot(x, comps['Peak3_']+comps['BackG_'], 'y--')
        if numPk==4:
            plt.plot(x, comps['Peak3_']+comps['BackG_'], 'y--')
            plt.plot(x, comps['Peak4_']+comps['BackG_'], 'g--')
        if numPk==5:
            plt.plot(x, comps['Peak3_']+comps['BackG_'], 'y--')
            plt.plot(x, comps['Peak4_']+comps['BackG_'], 'g--')   
            plt.plot(x, comps['Peak5_']+comps['BackG_'], 'm--')
        if numPk==6:
            plt.plot(x, comps['Peak3_']+comps['BackG_'], 'y--')
            plt.plot(x, comps['Peak4_']+comps['BackG_'], 'g--')   
            plt.plot(x, comps['Peak5_']+comps['BackG_'], 'm--')
            plt.plot(x, comps['Peak6_']+comps['BackG_'], 'k--')
        plt.plot(x, comps['BackG_'], 'r--')
        plt.title('Radial Intensity Distribution', fontsize=14)
        plt.xlabel('Angle of Orientation (degrees)', fontsize=14)
        plt.ylabel('Intensity (a.u.)', fontsize=14)
    #plt.show()
    
    yBestFit = out.best_fit - comps['BackG_']
    return out, yBestFit

    
def CosNth2D (x,y,Range='Pi'):
    if Range == '2Pi':
        div=1
    elif Range == 'Pi':
        div=2
    elif Range == 'Pi/2':
        div=4
    x=x[0:int(len(x)/div)]
    yAvg=(y[0:int(len(y)/div)]+y[int(len(y)/div)::])/2  # taking average of values upto Pi
    num=yAvg*np.power(np.cos(x*np.pi/180),2)
    den=yAvg
    cosSqrd2D=round(trapz(num,x)/trapz(den,x),3)
    num1=yAvg*np.power(np.cos(x*np.pi/180),4)
    cosFourth2D=round(trapz(num1,x)/trapz(den,x),3)
    return cosSqrd2D, cosFourth2D


def CosNth3D (x,y,Range='Pi'):
    #If range is 2Pi, then cos sqaured goes to negative due to sin term
    if Range == '2Pi':
        div=1
    elif Range == 'Pi':
        div=2
    elif Range == 'Pi/2':
        div=4
    x=x[0:int(len(x)/div)]
    y=(y[0:int(len(y)/div)]+y[int(len(y)/div)::])/2  # taking average of values upto Pi
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
    disp=DispImg.get()
    filtLevel=int(filtLev.get())
    smoothLevel=int(smoothLev.get())
    rotate=RotateImg.get()
    fit=fitReq.get()
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
    
    
        
    '''Image Analysis'''
    if selection==1:
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
            binImage = makebinary(processedImage)
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

        

        '''Image Plotting'''
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
        
    elif selection==2:
        data=readFile(filename)
        angle=data.angle0[1:].to_numpy()
        intensity=data.intensity0[1:].to_numpy()
        if rotate =='Yes':
            angle=angle-90
    
    
    plotRawdata= disp
    if plotRawdata=='Yes':
        plt.figure()
        plt.plot(angle,intensity,label='raw data')
        plt.title('Radial Intensity distribution', fontsize=14)
        plt.ylabel(r'Intensity, I($\theta$) (a.u.)', fontsize=14)
        plt.xlabel(r'Orientation, $\theta$ (degrees)',fontsize=14)
        plt.legend()
        plt.ion()
        plt.show()

        
    '''Calculation'''
    
    fitOutput=[]

    CosSq2Dsmth=[]
    CosQt2Dsmth=[]
    CosSq2Dfilt=[]
    CosQt2Dfilt=[]
    T2smth=[]
    T4smth=[]
    T2filt=[]
    T4filt=[]

    CosSq3Dsmth=[]
    CosQt3Dsmth=[]
    CosSq3Dfilt=[]
    CosQt3Dfilt=[]
    P2smth=[]
    P2filt=[]
    P4smth=[]
    P4filt=[]
    
       
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

        #smoothen the curve and calculate cosSqX etc
        smoothedIntensity=sp.savgol_filter(filteredIntensity,smoothLevel, 2)
        ySmooth=smoothedIntensity-min(smoothedIntensity)
        
        plt.figure()
        plt.plot(x,filteredIntensity,label='filtered')
        plt.plot(x,smoothedIntensity,label='smoothed')
        plt.title('Radial Intensity distribution', fontsize=14)
        plt.ylabel(r'Intensity, I($\theta$) (a.u.)', fontsize=14)
        plt.xlabel(r'Orientation, $\theta$ (degrees)',fontsize=14)
        plt.legend()
        #plt.show()
        
        
        #Here Alignment is orientation not wrt horizontal axis
        #but wrt to direction of max intensity
        
        maxAngS=x[ySmooth[0:len(ySmooth)//2].argmax()]
        if maxAngS>45 and maxAngS<=135:
            shift=90-maxAngS
        elif maxAngS>135 and maxAngS<=215:
            shift=180-maxAngS
        elif maxAngS<=45 and maxAngS>=0:
            shift=0-maxAngS
        xNew=x+shift
        print('Peak of smoothed curve is at',maxAngS,'degrees')
        print('Principal axis for smoothed peak shifted by',shift,'degrees')
        CosSq2Dsmth.append(CosNth2D(xNew,ySmooth)[0])
        CosQt2Dsmth.append(CosNth2D(xNew,ySmooth)[1])
        CosSq3Dsmth.append(CosNth3D(xNew,ySmooth)[0])
        CosQt3Dsmth.append(CosNth3D(xNew,ySmooth)[1])

                
        #fitting peaks to filteredintensity
        if fit=='Yes':
            fitOut,yFilter=fitModel(x,filteredIntensity,t1,t2,t3,t4,t5,t6,n,c1,c2,c3,c4,c5,c6)
            fitOutput.append(fitOut)
        elif fit=='No':
            yFilter = filteredIntensity
        yFilter=yFilter-min(yFilter)
        
               
        maxAngF=x[yFilter[0:len(yFilter)//2].argmax()]
        if maxAngF>45 and maxAngF<=135:
            shift=90-maxAngF
        elif maxAngF>135 and maxAngF<=215:
            shift=180-maxAngF
        elif maxAngF<=45 and maxAngF>=0:
            shift=0-maxAngF
        xNew=x+shift
        print('Peak of filtered curve is at',maxAngF,'degrees')
        print('Principal axis for filtered peak shifted by',shift,'degrees')
        CosSq2Dfilt.append(CosNth2D(xNew,yFilter)[0])
        CosQt2Dfilt.append(CosNth2D(xNew,yFilter)[1])
        CosSq3Dfilt.append(CosNth3D(xNew,yFilter)[0])
        CosQt3Dfilt.append(CosNth3D(xNew,yFilter)[1])

        print('The minimum value of smoothed and filtered ODFs are',min(ySmooth),'and',min(yFilter),'\n')

        plotShiftedODF='No'
        if plotShiftedODF=='Yes':
            plt.figure()
            plt.plot(xNew,filteredIntensity,label='filtered')
            plt.plot(xNew,smoothedIntensity,label='smoothed')
            plt.title('Shifted Radial Intensity distribution', fontsize=14)
            plt.ylabel(r'Intensity, I($\theta$) (a.u.)', fontsize=14)
            plt.xlabel(r'Orientation, $\theta$ (degrees)',fontsize=14)
            plt.legend()
            plt.show()
            
           
    for i in range(0,len(CosSq2Dsmth)):
        T2smth.append(ChebT2(CosSq2Dsmth[i]))
    for i in range(0,len(CosQt2Dsmth)):
        T4smth.append(ChebT4(CosSq2Dsmth[i],CosQt2Dsmth[i]))
    for i in range(0,len(CosSq3Dsmth)):
        P2smth.append(HermanP2(CosSq3Dsmth[i]))
    for i in range(0,len(CosQt3Dsmth)):
        P4smth.append(HermanP4(CosSq3Dsmth[i],CosQt3Dsmth[i]))
        

    for i in range(0,len(CosSq2Dfilt)):
        T2filt.append(ChebT2(CosSq2Dfilt[i]))
    for i in range(0,len(CosQt2Dfilt)):
        T4filt.append(ChebT4(CosSq2Dfilt[i],CosQt2Dfilt[i]))
    for i in range(0,len(CosSq3Dfilt)):
        P2filt.append(HermanP2(CosSq3Dfilt[i]))
    for i in range(0,len(CosQt3Dfilt)):
        P4filt.append(HermanP4(CosSq3Dfilt[i],CosQt3Dfilt[i]))
        
 
        
    CosSq2DsmthMean=np.round(np.mean(CosSq2Dsmth),3)
    CosSq2DsmthDev=np.round(np.std(CosSq2Dsmth),3)
    CosQt2DsmthMean=np.round(np.mean(CosQt2Dsmth),3)
    CosQt2DsmthDev=np.round(np.std(CosQt2Dsmth),3)
    T2smthAvg=np.round(np.mean(T2smth),3)
    T2smthDev=np.round(np.std(T2smth),3)
    T4smthAvg=np.round(np.mean(T4smth),3)
    T4smthDev=np.round(np.std(T4smth),3)

    CosSq2DfiltMean=np.round(np.mean(CosSq2Dfilt),3)
    CosSq2DfiltDev=np.round(np.std(CosSq2Dfilt),3)
    CosQt2DfiltMean=np.round(np.mean(CosQt2Dfilt),3)
    CosQt2DfiltDev=np.round(np.std(CosQt2Dfilt),3)
    T2filtAvg=np.round(np.mean(T2filt),3)
    T2filtDev=np.round(np.std(T2filt),3)
    T4filtAvg=np.round(np.mean(T4filt),3)
    T4filtDev=np.round(np.std(T4filt),3)

    CosSq3DsmthMean=np.round(np.mean(CosSq3Dsmth),3)
    CosSq3DsmthDev=np.round(np.std(CosSq3Dsmth),3)
    CosQt3DsmthMean=np.round(np.mean(CosQt3Dsmth),3)
    CosQt3DsmthDev=np.round(np.std(CosQt3Dsmth),3)
    P2smthAvg=np.round(np.mean(P2smth),3)
    P2smthDev=np.round(np.std(P2smth),3)
    P4smthAvg=np.round(np.mean(P4smth),3)
    P4smthDev=np.round(np.std(P4smth),3)

    CosSq3DfiltMean=np.round(np.mean(CosSq3Dfilt),3)
    CosSq3DfiltDev=np.round(np.std(CosSq3Dfilt),3)
    CosQt3DfiltMean=np.round(np.mean(CosQt3Dfilt),3)
    CosQt3DfiltDev=np.round(np.std(CosQt3Dfilt),3)
    P2filtAvg=np.round(np.mean(P2filt),3)
    P2filtDev=np.round(np.std(P2filt),3)
    P4filtAvg=np.round(np.mean(P4filt),3)
    P4filtDev=np.round(np.std(P4filt),3)
    
       
    
    print ('\nThe Orientation Parameters w.r.t the principal axis for')
    print(' smoothed data (no fitting) are')
    #print('\t cos^2Theta =', CosSq2Dsmth,', Average cos^2Theta =', CosSq2DsmthMean,'+/-',CosSq2DsmthDev)
    #print('\t cos^4Theta =', CosQt2Dsmth,', Average cos^4Theta =', CosQt2DsmthMean,'+/-',CosQt2DsmthDev)
    #print('\t cos^2Theta =', CosSq3Dsmth,', Average cos^2Theta =', CosSq3DsmthMean,'+/-',CosSq3DsmthDev)
    #print('\t cos^4Theta =', CosQt3Dsmth,', Average cos^4Theta =', CosQt3DsmthMean,'+/-',CosQt3DsmthDev)
    print('\t Chebyshev T2 =',T2smth,', Average T2 =', T2smthAvg,'+/-',T2smthDev)
    print('\t Chebyshev T4 =',T4smth,', Average T4 =', T4smthAvg,'+/-',T4smthDev)
    print('\t Hermann P2 =',P2smth,', Average P2 =', P2smthAvg,'+/-',P2smthDev)
    print('\t Hermann P4 =',P4smth,', Average P4 =', P4smthAvg,'+/-',P4smthDev)
    
    
    
    if fit =='Yes':
        print('\n filtered data (after fitting) are')
        #print('\t cos^2Theta =', CosSq2Dfilt,', Average cos^2Theta =', CosSq2DfiltMean,'+/-',CosSq2DfiltDev)
        #print('\t cos^4Theta =', CosQt2Dfilt,', Average cos^4Theta =', CosQt2DfiltMean,'+/-',CosQt2DfiltDev)
        #print('\t cos^2Theta =', CosSq3Dfilt,', Average cos^2Theta =', CosSq3DfiltMean,'+/-',CosSq3DfiltDev)
        #print('\t cos^3Theta =', CosQt3Dfilt,', Average cos^4Theta =', CosQt3DfiltMean,'+/-',CosQt3DfiltDev)
        print('\t Chebyshev T2 =',T2filt,', Average T2 =', T2filtAvg,'+/-',T2filtDev)
        print('\t Chebyshev T4 =',T4filt,', Average T4 =', T4filtAvg,'+/-',T4filtDev)
        print('\t Hermann P2 =',P2filt,', Average P2 =', P2filtAvg,'+/-',P2filtDev)
        print('\t Hermann P4 =',P4filt,', Average P4 =', P4filtAvg,'+/-',P4filtDev)
        
       
    else:
        print('\n filtered data (without fitting) are')
        #print('\t cos^2Theta =', CosSq2Dfilt,', Average cos^2Theta =', CosSq2DfiltMean,'+/-',CosSq2DfiltDev)
        #print('\t cos^4Theta =', CosQt2Dfilt,', Average cos^4Theta =', CosQt2DfiltMean,'+/-',CosQt2DfiltDev)
        #print('\t cos^2Theta =', CosSq3Dfilt,', Average cos^2Theta =', CosSq3DfiltMean,'+/-',CosSq3DfiltDev)
        #print('\t cos^3Theta =', CosQt3Dfilt,', Average cos^4Theta =', CosQt3DfiltMean,'+/-',CosQt3DfiltDev)
        print('\t Chebyshev T2 =',T2filt,', Average T2 =', T2filtAvg,'+/-',T2filtDev)
        print('\t Chebyshev T4 =',T4filt,', Average T4 =', T4filtAvg,'+/-',T4filtDev)
        print('\t Hermann P2 =',P2filt,', Average P2 =', P2filtAvg,'+/-',P2filtDev)
        print('\t Hermann P4 =',P4filt,', Average P4 =', P4filtAvg,'+/-',P4filtDev)
        
        
    output=np.asarray([T2smth,T4smth,P2smth,P4smth,T2filt,T4filt,P2smth,P4filt])
    output=output.T
    #np.savetxt('Chebdata.txt',output,fmt='%f',delimiter=',',header='T2(smooth),T4(smooth),P2(smooth),P4(smooth),T2(filetered),T4(filetered),P2(filetered),P4(filetered)')    
    C=('T2(smooth)','T4(smooth)','P2(smooth)','P4(smooth)','T2(filetered)','T4(filetered)','P2(filetered)','P4(filetered)')
    pd.DataFrame(output).to_csv("FibreCOP_result.csv", index=False,header=C)


window = tk.Tk() #Create window object

window.title('FibreCOP:Chebyshev Orientation Parameter for CNT textiles')
window.geometry("560x780")

label1 = tk.Label(window, text=" Enter File Path")
label1.grid(row=0,column=1, pady=5)

filePath=tk.StringVar()
entry1=tk.Entry(window,width=40,textvariable=filePath)
entry1.grid(row=0,column=2,columnspan=3, padx=5, pady=5)

browseButton = tk.Button(window, text='Browse', command=openFile)
browseButton.grid(row=0,column=5,sticky=tk.EW, pady=5)

label56 = tk.Label(window, text="Data Type") 
label56.grid(row=1,column=1,padx=5)
choice=tk.IntVar()
radio1=tk.Radiobutton(window,text="Image",variable=choice,value=1)
radio1.grid(row=1,column=2,sticky=tk.W,padx=5)
radio2=tk.Radiobutton(window,text="Data",variable=choice,value=2)
radio2.grid(row=1,column=3,sticky=tk.W,padx=5)

can = tk.Canvas(window, bg="white", height=80, width=80,relief='sunken')
can.grid(row=0,rowspan=5,column=0)

imfile = tk.PhotoImage(file = "icon.gif")
imfile= imfile.subsample(3,3)
image = can.create_image(40,40, anchor=tk.CENTER, image=imfile)

label2 = tk.Label(window, text="Image Analysis Options", font=('Helevetica',13)) 
label2.grid(row=4,column=1,columnspan=2, padx=5, pady=5)

label4 = tk.Label(window, text="Strip height") 
label4.grid(row=6,column=0, sticky=tk.W,padx=5, pady=5)
stripHeight=tk.StringVar()
stripHeight.set("7")
entry4=tk.Entry(window,width=20,textvariable=stripHeight)
entry4.grid(row=6,column=1,pady=5)
label14=tk.Label(window, text="(height in % of SEM info bar to be stripped)")
label14.grid(row=6,column=2, columnspan=4,sticky=tk.W,padx=5, pady=5)

labe57 = tk.Label(window, text="Rotate Image") 
labe57.grid(row=7,column=0, sticky=tk.W,padx=5, pady=5)
RotateImg=tk.StringVar()
RotateImg.set("Yes")
option57=tk.OptionMenu(window,RotateImg,"Yes","No")
option57.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option57.grid(row=7,column=1,pady=5)
label57=tk.Label(window, text="(rotate by 90 if image horizontal to get +ve OP)")
label57.grid(row=7,column=2,columnspan=4, sticky=tk.W,padx=5, pady=5)

label5 = tk.Label(window, text="De-noising") 
label5.grid(row=8,column=0, sticky=tk.W,padx=5, pady=5)
Denoise=tk.StringVar()
Denoise.set("0")
entry5=tk.Entry(window,width=20,textvariable=Denoise)
entry5.grid(row=8,column=1,pady=5)
label15=tk.Label(window, text="(increases from 0 to 1, for noisy images use ~ 0.5)")
label15.grid(row=8,column=2, columnspan=4,sticky=tk.W,padx=5, pady=5)

label6 = tk.Label(window, text="No. of Scans") 
label6.grid(row=9,column=0, sticky=tk.W,padx=5, pady=5)
xScan=tk.StringVar()
xScan.set("1")
entry6=tk.Entry(window,width=20,textvariable=xScan)
entry6.grid(row=9,column=1,pady=5)
label16=tk.Label(window, text="(no. of square areas to be scanned,> 0; for data use 1)")
label16.grid(row=9,column=2,columnspan=4, sticky=tk.W,padx=5, pady=5)

label7 = tk.Label(window, text="Bin Size") 
label7.grid(row=11,column=0, sticky=tk.W,padx=5, pady=5)
BinSize=tk.StringVar()
BinSize.set("0.25")
entry7=tk.Entry(window,width=20,textvariable=BinSize)
entry7.grid(row=11,column=1,pady=5)
label17=tk.Label(window, text="(< 1, sector angle for radial summation)")
label17.grid(row=11,column=2,columnspan=4, sticky=tk.W,padx=5, pady=5)

label45 = tk.Label(window, text="Display Images") 
label45.grid(row=12,column=0, sticky=tk.W,padx=5, pady=5)
DispImg=tk.StringVar()
DispImg.set("Yes")
option45=tk.OptionMenu(window,DispImg,"Yes","No")
option45.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option45.grid(row=12,column=1,pady=5)
label145=tk.Label(window, text="(display images used for analysis)")
label145.grid(row=12,column=2,columnspan=4, sticky=tk.W,padx=5, pady=5)

labe20 = tk.Label(window, text="Filter Interval") 
labe20.grid(row=14,column=0, sticky=tk.W,padx=5, pady=5)
filtLev=tk.StringVar()
filtLev.set("5")
entry20=tk.Entry(window,width=20,textvariable=filtLev)
entry20.grid(row=14,column=1,pady=5)
label20=tk.Label(window, text="(>=3, odd, window size for median filter)")
label20.grid(row=14,column=2,columnspan=4, sticky=tk.W,padx=5, pady=5)

labe21 = tk.Label(window, text="Smoothing") 
labe21.grid(row=15,column=0, sticky=tk.W,padx=5, pady=5)
smoothLev=tk.StringVar()
smoothLev.set("51")
entry21=tk.Entry(window,width=20,textvariable=smoothLev)
entry21.grid(row=15,column=1,pady=5)
label21=tk.Label(window, text="(>=3 , odd, window size for Savitzky Golay)")
label21.grid(row=15,column=2,columnspan=4, sticky=tk.W,padx=5, pady=5)

labe47 = tk.Label(window, text="Peak Fitting Options", font=('Helevetica',13)) 
labe47.grid(row=16,column=1,columnspan=2, padx=5, pady=5)

label9 = tk.Label(window, text="Fitting Required") 
label9.grid(row=17,column=0, sticky=tk.W,padx=5, pady=5)
fitReq=tk.StringVar()
fitReq.set("Yes")
option9=tk.OptionMenu(window,fitReq,"Yes","No")
option9.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option9.grid(row=17,column=1,pady=5)
label19=tk.Label(window, text="(tip: sharp peaks with wider base need extra Gaussian)")
label19.grid(row=17,column=2,columnspan=4, sticky=tk.W,padx=5, pady=5)

label29 = tk.Label(window, text="No. of Peaks") 
label29.grid(row=18,column=0, sticky=tk.W,padx=5, pady=5)
noPk=tk.StringVar()
noPk.set("3")
option29=tk.OptionMenu(window,noPk,"2","3","4","5","6")
option29.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option29.grid(row=18,column=1,pady=5)
label49=tk.Label(window, text="(2<=n<=6, min. vertical=3, horizontal=2)")
label49.grid(row=18,column=2,columnspan=4, sticky=tk.W,padx=5, pady=5)


label22 = tk.Label(window, text="Peak 1") 
label22.grid(row=19,column=0, sticky=tk.EW,padx=5, pady=5)
fitTyp1=tk.StringVar()
fitTyp1.set("Lorentzian")
option22=tk.OptionMenu(window,fitTyp1,"Lorentzian","Gaussian","PseudoVoigt")
option22.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option22.grid(row=19,column=1,pady=5, padx=5)


label23 = tk.Label(window, text="Peak 2") 
label23.grid(row=20,column=0, sticky=tk.EW,padx=5, pady=5)
fitTyp2=tk.StringVar()
fitTyp2.set("Lorentzian")
option23=tk.OptionMenu(window,fitTyp2,"Lorentzian","Gaussian","PseudoVoigt")
option23.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option23.grid(row=20,column=1,pady=5,padx=5)

label24 = tk.Label(window, text="Peak 3") 
label24.grid(row=21,column=0, sticky=tk.EW,padx=5, pady=5)
fitTyp3=tk.StringVar()
fitTyp3.set("Lorentzian")
option24=tk.OptionMenu(window,fitTyp3,"Lorentzian","Gaussian","PseudoVoigt")
option24.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option24.grid(row=21,column=1,pady=5,padx=5)


label25 = tk.Label(window, text="Peak 4") 
label25.grid(row=22,column=0, sticky=tk.EW,padx=5, pady=5)
fitTyp4=tk.StringVar()
fitTyp4.set("Lorentzian")
option25=tk.OptionMenu(window,fitTyp4,"Lorentzian","Gaussian","PseudoVoigt")
option25.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option25.grid(row=22,column=1,pady=5,padx=5)

label26 = tk.Label(window, text="Peak 5") 
label26.grid(row=23,column=0, sticky=tk.EW,padx=5, pady=5)
fitTyp5=tk.StringVar()
fitTyp5.set("Lorentzian")
option26=tk.OptionMenu(window,fitTyp5,"Lorentzian","Gaussian","PseudoVoigt")
option26.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option26.grid(row=23,column=1,pady=5,padx=5)


label27 = tk.Label(window, text="Peak 6") 
label27.grid(row=24,column=0, sticky=tk.EW,padx=5, pady=5)
fitTyp6=tk.StringVar()
fitTyp6.set("Lorentzian")
option27=tk.OptionMenu(window,fitTyp6,"Lorentzian","Gaussian","PseudoVoigt")
option27.configure(width=14,bg='white',bd=1,activebackground='white',relief='sunken')
option27.grid(row=24,column=1,pady=5, padx=5)

label32 = tk.Label(window, text="Centre") 
label32.grid(row=19,column=2,sticky=tk.E, pady=5)
cen1=tk.StringVar()
cen1.set("1")
entry32=tk.Entry(window,width=10,textvariable=cen1)
entry32.grid(row=19,column=3,pady=5)

label33 = tk.Label(window, text="Centre") 
label33.grid(row=20,column=2,sticky=tk.E, pady=5)
cen2=tk.StringVar()
cen2.set("180")
entry33=tk.Entry(window,width=10,textvariable=cen2)
entry33.grid(row=20,column=3,pady=5)

label34 = tk.Label(window, text="Centre") 
label34.grid(row=21,column=2,sticky=tk.E, pady=5)
cen3=tk.StringVar()
cen3.set("359")
entry34=tk.Entry(window,width=10,textvariable=cen3)
entry34.grid(row=21,column=3,pady=5)

label35 = tk.Label(window, text="Centre") 
label35.grid(row=22,column=2,sticky=tk.E, pady=5)
cen4=tk.StringVar()
cen4.set("0")
entry35=tk.Entry(window,width=10,textvariable=cen4)
entry35.grid(row=22,column=3,pady=5)

label36 = tk.Label(window, text="Centre") 
label36.grid(row=23,column=2,sticky=tk.E, pady=5)
cen5=tk.StringVar()
cen5.set("0")
entry36=tk.Entry(window,width=10,textvariable=cen5)
entry36.grid(row=23,column=3,pady=5)

label37= tk.Label(window, text="Centre") 
label37.grid(row=24,column=2,sticky=tk.E, pady=5)
cen6=tk.StringVar()
cen6.set("0")
entry37=tk.Entry(window,width=10,textvariable=cen6)
entry37.grid(row=24,column=3,pady=5)

'''
label24 = tk.Label(window, text="Chebyshev Options", font=('Helevetica',13)) 
label24.grid(row=24,column=0,columnspan=3, padx=5, pady=10)

'''
calcButton = tk.Button(window, text='Calculate COP', command=calcCOP)
calcButton.grid(row=100,column=1, columnspan = 3, sticky= tk.EW, padx=5, pady=10)

closeButton = tk.Button(window, text='Close Graphs', command=destroyWindows)
closeButton.grid(row=100,column=0, columnspan=1, sticky= tk.EW, padx=5, pady=10)

#saveButton = tk.Button(window, text="Save",command=saveText) 
#saveButton.grid(row=110,column=1,sticky=tk.EW,padx=5, pady=10)

quitButton = tk.Button(window, text="Quit",command=quitProgram) 
quitButton.grid(row=100,column=4,columnspan=2,sticky=tk.EW,padx=5, pady=10)

label72= tk.Label(window, text="@ AK2011, Macromolecular Materials Laboratory, University of Cambridge, 2020", font=('Helevetica',7)) 
label72.grid(row=101,column=0,columnspan=5,sticky=tk.EW, padx=5)

window.mainloop()



''' Program Ends Here'''

'''
Improvements to be done
1. diagonally (or at 180 degrees) oriented images do not fit well
2. allow users to chose peak parameters


'''
