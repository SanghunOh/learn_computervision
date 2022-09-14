import cv2
import numpy as np

img = cv2.imread('Paris.jpg')
height, width = img.shape[:2]
y = np.ones((height, width), np.uint8) * 128
output = np.zeros((height, width), np.uint8)
# generating the kernels
kernel1 = np.array([[0, -1, -1], # kernel for embossing bottom left side
                    [1, 0, -1],
                    [1, 1, 0]])
kernel2 = np.array([[-1, -1, 0], # kernel for embossing bottom right side
                    [-1, 0, 1],
                    [0, 1, 1]])
# you can generate kernels for embossing top as well
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
output1 = cv2.add(cv2.filter2D(gray, -1, kernel1), y) # emboss on bottom left side
output2 = cv2.add(cv2.filter2D(gray, -1, kernel2), y) # emboss on bottom right side
for i in range(height):
    for j in range(width):
        output[i, j] = max(output1[i, j], output2[i, j]) # combining both embosses to produce stronger emboss
cv2.imshow("Image", img)
cv2.imshow("Output1", output1)
cv2.imshow("Output2", output2)
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()