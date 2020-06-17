import cv2
import numpy as np
import matplotlib.pyplot as plt


class MutualInformation:

    bin_size = 256  #Default value of the bin_size

    def __init__(self, image):
        self.image = image
        self.h = self.getShape(image)[0]
        self.w = self.getShape(image)[1]
    
    def setBinSize(self, size):
        """
        The method assigns value to bin_size
        :param word: size - Value of the bin_Size
        """
        self.bin_size = size
    
    def getShape(self,image):
        """
        The method returns shape of the matrix
        :param word: image - Input image
        """
        return np.shape(image)
    
    def splitChannels(self):
        """
        The method seperates b,g,r channels as 3 separate gray scale images
        and returns them
        """
        blue,green,red = cv2.split(self.image)
        return  blue,green,red

    def coloredChannels(self, blue, green, red):
        """
        The method returns seperated b,g,r channels as 3 separate colored images
        :param word: blue - Blue channel
        :param word: green - Green channel
        :param word: red - Red channel
        """
        zeros = np.zeros(blue.shape, np.uint8)

        blueBGR = cv2.merge((blue,zeros,zeros))
        greenBGR = cv2.merge((zeros,green,zeros))
        redBGR = cv2.merge((zeros,zeros,red))

        return blueBGR,greenBGR,redBGR
    
    def cropImage(self, image, x, x1 ,y, y1, h, w):
        """
        The method crops the image by sepcified dimenstions and 
        then returns it
        :param word: image - Input image
        :param word: x,x1 - Value of x,x1 
        :param word: y,y1 - Value of y,y1 
        :param word: h - Height of the image
        :param word: w - Width of the image
        """
        cropedImg = image[y:y1+h, x:x1+w].copy()
        return cropedImg

    def getArray(self, image):
        """
        The method converts matrix into 1d array and
        return it
        :param word: image - Input image
        """
        image1D = np.ravel(image)
        return image1D

    def entropy(self,hist):
        """
        The method calculates entropy of the image and returns it
        :param word: hist - Histogram of the image
        """
        dataNormalized = hist[0]/float(np.sum(hist[0]))
        noneZeroData= dataNormalized[dataNormalized != 0]
        ent = -(noneZeroData*np.log(np.abs(noneZeroData))).sum() 
        return ent

    def mutualInformation(self, img1, img2):
        """
        The method calculates mutual information of the two images 
        (according to I(X:Y):= H(X) + H(Y) − H(XY) 
        with the Shannon entropy of "X" and "Y" >> H(X), H(Y), 
        and the Shannon entropy of the pair "(X,Y)" >> H(XY))
        and returns it
        :param word: img1 -  1D array of the first image values
        :param word: ima2 - 1D array of the second image values
        """
        HistX = np.histogram(img1, bins= self.bin_size, range=(0,256), density=True)
        HistY = np.histogram(img2, bins= self.bin_size, range=(0,256), density=True)
        HistXY = np.histogram2d(img1,img2, bins= self.bin_size)

        entX = self.entropy(HistX)
        entY = self.entropy(HistY)
        jointEntXY = self.entropy(HistXY)
    
        return (entX + entY - jointEntXY)

    def getMIPlot(self, x, y, fileName):
        """
        The method plots the mutual information as a function of 
        the x-position of the red channel image and saves it
        :param word: x - Array of red channel x-positions per iteration 
        :param word: y - Array of MI values per iteration
        """
        plt.plot(x, y)
        plt.savefig('../output/'+fileName+'_MI_plot.pdf')

        plt.scatter(x, y)
        plt.savefig('../output/'+fileName+'_MI_scattered.pdf')

userInput = input()
image = cv2.imread('../input/'+userInput+'.png',1)  
mi = MutualInformation(image)

#Separating blue, green and red channels as gray scale images
b,g,r = mi.splitChannels()
#Cropping green channel image from the left
green = mi.cropImage(g,20,20,0,0,mi.h, mi.w)
green_h,green_w = mi.getShape(green)
#Cropping green channel image from the right
green = mi.cropImage(green,0,-20,0,0,green_h, green_w)
green_h,green_w = mi.getShape(green)
green_size = green_h * green_w

# Converting matrix to 1D array
green_Array = mi.getArray(green)
# Setting bin_size
# mi.setBinSize()

mi_array = []
directionX = 0
#Moving the red channel image in x-direction (in 41 steps from left to right)
while directionX != 41:
    #Getting red channel values
    #  of the overlapping region
    red_pixels = (np.mgrid[0:green_h, directionX: directionX + green_w]).reshape(2, -1).T
    red_Array = []
    for x in red_pixels:
        red_Array.append(r[x[0]][x[1]])
    #Getting int 1D array
    red_Array = mi.getArray(red_Array)

    #Calculating Mutual Information of the overlapping region
    mi_res = mi.mutualInformation(green_Array,red_Array)
    mi_array.append(mi_res)

    directionX = directionX + 1

arr_iterations = np.arange(0, 41)
#Plotting the graphs
mi.getMIPlot(arr_iterations, mi_array,userInput)

cv2.imwrite('../output/' + userInput + '_green.png',g)
cv2.imwrite('../output/' + userInput + '_red.png',r)
cv2.imwrite('../output/' + userInput + '_green_cropped.png',green)

# # Writing seperated channel images as colored images
# bc,gc,rc = mi.coloredChannels(b,g,r)
# cv2.imwrite('../output/' + userInput + '_green_colored.png',gc)
# cv2.imwrite('../output/' + userInput + '_red_colored.png',rc)
