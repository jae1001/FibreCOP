import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

radius = 6              # exclusion zone around points (should be even)
box = 200               # limit on size of bounding box (cubic)
qty = 200               # number of points in box
length = int(radius/2)  # length of "fibre"
zbias = 10               # zbias factor to control orientation - 1 is uniform, higher is more z-oriented

rangeX = (0, box)
rangeY = (0, box)
rangeZ = (0, box)

# carry out 3D FFT
def TwoDFourierTrans(image):
    #Perform 2DFourier transform
    #https://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
    fftImage=np.fft.fft2(image)
    fftShiftImage=np.fft.fftshift(fftImage)
    fftMagImage=np.abs(fftShiftImage)
    #fftMagImage=np.log(1+fftMagImage)
    (h,w)=fftMagImage.shape
    return fftMagImage,h,w

# carry out 3D FFT
def ThreeDFourierTrans(image):
    #Perform 3D Fourier transform
    #https://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
    fftImage=np.fft.fftn(image)
    fftShiftImage=np.fft.fftshift(fftImage)
    fftMagImage=np.abs(fftShiftImage)
    #fftMagImage=np.log(1+fftMagImage)
    (h,w,d)=fftMagImage.shape
    return fftMagImage,h,w,d

# flatten nested lists
def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

# define random vector on unit sphere
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec[2] *= zbias
    vec /= np.linalg.norm(vec, axis=0)
    return vec

# generate a set of quasi-randomly distributed points within bounding box which
# have a minimum separation "radius"

# Generate a set of all points within "radius" of the origin, to be used as offsets later
# There's probably a more efficient way to do this.
deltas = set()
for x in range(-radius, radius+1):
    for y in range(-radius, radius+1):
        for z in range(-radius, radius+1):
           if x*x + y*y + z*z <= radius*radius:
               deltas.add((x,y,z))

randPoints = []
excluded = set()
i = 0
while i<qty:
    x = random.randrange(*rangeX)
    y = random.randrange(*rangeY)
    z = random.randrange(*rangeZ)
    if (x,y,z) in excluded: continue
    randPoints.append((x,y,z))
    i += 1
    excluded.update((x+dx, y+dy, z+dz) for (dx,dy,dz) in deltas)
excluded.clear()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# extend each point into a "fibre" with length "length"

rodPoints = []
for point in randPoints:
    randsphere = sample_spherical(1)
    for x in range (-length*10, (length+1)*10):
        vec = randsphere*x/2
        vec = np.rint(flatten(vec.tolist())+np.array(point))
        rodPoints.append(vec)

ax.scatter(*zip(*randPoints))
ax.scatter(*zip(*rodPoints))
plt.ion()
plt.show()

# convert list of points into a floating point array, implementing
# periodic boundary conditions on the edge of box

image = np.zeros((box,box,box),dtype='f4')

for point in rodPoints:
    x = np.array(point)[0]
    y = np.array(point)[1]
    z = np.array(point)[2]
    if x>=box : x -= box
    if y>=box : y -= box
    if z>=box : z -= box
    if x<0 : x += box
    if y<0 : y += box
    if z<0 : z += box
    
    image[int(x),int(y),int(z)] = 1.0

#   Now thicken the fibres by additing additional points around each pixel

#   Fill the face sites
    image[int((x+1)%box),int(y),int(z)] = 1.0
    image[int((x-1)%box),int(y),int(z)] = 1.0
    image[int(x),int((y+1)%box),int(z)] = 1.0
    image[int(x),int((y-1)%box),int(z)] = 1.0
    image[int(x),int(y),int((z+1)%box)] = 1.0
    image[int(x),int(y),int((z-1)%box)] = 1.0

#   Fill the edge sites
    image[int((x+1)%box),int((y+1)%box),int(z)] = 1.0
    image[int((x-1)%box),int((y-1)%box),int(z)] = 1.0
    image[int((x+1)%box),int((y-1)%box),int(z)] = 1.0
    image[int((x-1)%box),int((y+1)%box),int(z)] = 1.0

    image[int((x+1)%box),int(y),int((z+1)%box)] = 1.0
    image[int((x-1)%box),int(y),int((z-1)%box)] = 1.0
    image[int((x+1)%box),int(y),int((z-1)%box)] = 1.0
    image[int((x-1)%box),int(y),int((z+1)%box)] = 1.0

    image[int(x),int((y+1)%box),int((z+1)%box)] = 1.0
    image[int(x),int((y-1)%box),int((z-1)%box)] = 1.0
    image[int(x),int((y+1)%box),int((z-1)%box)] = 1.0
    image[int(x),int((y-1)%box),int((z+1)%box)] = 1.0

#   Fill the corner sites

    image[int((x+1)%box),int((y+1)%box),int((z+1)%box)] = 1.0
    image[int((x-1)%box),int((y+1)%box),int((z+1)%box)] = 1.0
    image[int((x+1)%box),int((y-1)%box),int((z+1)%box)] = 1.0
    image[int((x-1)%box),int((y-1)%box),int((z+1)%box)] = 1.0
    image[int((x+1)%box),int((y-1)%box),int((z-1)%box)] = 1.0
    image[int((x-1)%box),int((y-1)%box),int((z-1)%box)] = 1.0

# display 2D projections of array down x, y and z-axes

f , axarr = plt.subplots(3,3,figsize=(8,8))
axarr[0,0].imshow(np.sum(image,axis=0),vmin=0, vmax=1)
axarr[0,1].imshow(np.sum(image,axis=1),vmin=0, vmax=1)
axarr[0,2].imshow(np.sum(image,axis=2),vmin=0, vmax=1)
plt.gray()

# display Fourier transforms of 2D projections

xx,box,box = TwoDFourierTrans(np.sum(image,axis=0))
yy,box,box = TwoDFourierTrans(np.sum(image,axis=1))
zz,box,box = TwoDFourierTrans(np.sum(image,axis=2))

axarr[1,0].imshow(xx, norm=LogNorm())
axarr[1,1].imshow(yy, norm=LogNorm())
axarr[1,2].imshow(zz, norm=LogNorm())
plt.gray()

# display slices of 3D Fourier transform

iimage,box,box,box = ThreeDFourierTrans(image)

axarr[2,0].imshow(iimage[int(box/2),:,:], norm=LogNorm())
axarr[2,1].imshow(iimage[:,int(box/2),:], norm=LogNorm())
axarr[2,2].imshow(iimage[:,:,int(box/2)], norm=LogNorm())
plt.gray()

plt.ion()
plt.show()

