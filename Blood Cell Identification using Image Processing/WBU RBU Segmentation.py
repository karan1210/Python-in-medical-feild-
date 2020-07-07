import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

# read original image 
image = cv2.imread("c120.png") 

# convet to gray scale image 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 


# apply median filter for smoothning 
blurM = cv2.medianBlur(gray, 5) 


# apply gaussian filter for smoothning 
blurG = cv2.GaussianBlur(gray, (9, 9), 0) 

# histogram equalization 
histoNorm = cv2.equalizeHist(gray) 


# create a CLAHE object for contrast Limited Adaptive Histogram Equalization (CLAHE) 
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8)) 
claheNorm = clahe.apply(gray) 



# contrast stretching 
# Function to map each intensity level to output intensity level. 
def pixelVal(pix, r1, s1, r2, s2): 
	if (0 <= pix and pix <= r1): 
		return (s1 / r1) * pix 
	elif (r1 < pix and pix <= r2): 
		return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1 
	else: 
		return ((255 - s2) / (255 - r2)) * (pix - r2) + s2 

# Define parameters.

r1 = 70
s1 = 0
r2 = 200
s2 = 255

# Vectorize the function to apply it to each value in the Numpy array. 
pixelVal_vec = np.vectorize(pixelVal) 

# Apply contrast stretching. 
contrast_stretched = pixelVal_vec(gray, r1, s1, r2, s2) 
contrast_stretched_blurM = pixelVal_vec(blurM, r1, s1, r2, s2)
plt.imshow(contrast_stretched_blurM)
plt.show()
cv2.imwrite('Final_Output_colour.png', contrast_stretched_blurM) 

# edge detection using canny edge detector 
edge = cv2.Canny(gray, 100, 200) 

edgeG = cv2.Canny(blurG, 100, 200) 

edgeM = cv2.Canny(blurM, 100, 200) 
cv2.imwrite('Final_Output.png', edgeM) 


#Plotting Images
titles = ['Original img','gray','bliurM','blurG','HistEq','Contrast Limited Adaptive Histogram Equalization','contrast_stretched','contrast_stretched_blurM','canny edge detector','cannyBlurG','CannnyBlurM']
images = [image,gray,blurM,blurG, histoNorm,claheNorm ,contrast_stretched,contrast_stretched_blurM,edge,edgeG,edgeM ]
for i in range(len(titles)):
    plt.subplot(4,3,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
