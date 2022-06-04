# code for displaying multiple images in one figure

#import libraries
import cv2 
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(10, 7))

# setting values to rows and column variables
rows = 2
columns = 3

# reading images
Image1 = cv2.imread('pol_bounds_1.png')
Image2 = cv2.imread('pol_bounds_2.png')
Image3 = cv2.imread('pol_bounds_3.png')
Image4 = cv2.imread('pol_bounds_4.png')
Image5 = cv2.imread('pol_bounds_5.png')

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(Image1)
plt.axis('off')

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(Image2)
plt.axis('off')

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(Image3)
plt.axis('off')

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)

# showing image
plt.imshow(Image4)
plt.axis('off')

#Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 5)

# showing image
plt.imshow(Image4)
plt.axis('off')

plt.show()