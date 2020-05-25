import cv2
import numpy as np

def skin_seg(img, bounding_boxes):
    #https://github.com/Jeanvit/PySkinDetection
    seg_img = np.copy(img)
    
    premask = np.zeros((seg_img.shape[0:2]))
    for (x,y,w,h) in bounding_boxes:
            premask[y:y+h, x:x+w] = 255
    premask_bool = (premask == 0)
    seg_img[premask_bool] = 0
    
    #cv2.imshow('debug', seg_img)
    #cv2.waitKey(0)

    img_YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_HSV_values = np.array([0, 40, 0], dtype = "uint8")
    upper_HSV_values = np.array([25, 255, 255], dtype = "uint8")

    lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
    upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

    #A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
    mask_YCbCr = cv2.inRange(img_YCbCr, lower_YCbCr_values, upper_YCbCr_values)
    mask_HSV = cv2.inRange(img_HSV, lower_HSV_values, upper_HSV_values) 

    binary_mask_image = cv2.add(mask_HSV,mask_YCbCr)
    mask_bool = (binary_mask_image == 0)
    seg_img[mask_bool] = 0
    #seg_img returns original img but values outside of threshold are zeroed(become black)
    #binary_mask_image returns a mask of 0s (no skin) and 255s (skin)
    return seg_img, binary_mask_image 

