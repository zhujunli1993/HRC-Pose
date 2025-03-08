import numpy as np
import cv2

print("labels: ",np.load("2_labels.npy"))
print("boxes: ",np.load("2_boxes.npy"))
data=np.load("2_masks.npy")
print("masks shape: ",data.shape)
print("image shape: ",cv2.imread("2.jpg").shape)
np.savetxt("mask.txt", data[0][0])  # Save as a .txt file
