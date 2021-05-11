import numpy as np
import cv2
import os
from PIL import Image, ImageFont, ImageDraw

name = 'frame_171.bmp'

jpeg_folder = 'experiments/jpeg_compression'
neural_folder = 'experiments/neural'
svd_folder = 'experiments/svd50'
correct_folder = 'correct'
jpeg = cv2.imread(os.path.join(jpeg_folder, name[:-4] + '.jpeg'))
neural = cv2.imread(os.path.join(neural_folder, name))
neural = cv2.resize(neural, (1280, 720))
svd = cv2.imread(os.path.join(svd_folder, name))
correct = cv2.imread(os.path.join(correct_folder, name))

img_combine1 = np.concatenate((correct, jpeg), axis=1)
img_combine2 = np.concatenate((neural, svd), axis=1)
img_combine = np.concatenate((img_combine1, img_combine2), axis=0)
print(img_combine.shape)
cv2.imwrite(os.path.join('photos', name[:-4] + '.png'), img_combine)
img = Image.open(os.path.join('photos', name[:-4] + '.png'))
image_editable = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 120)
image_editable.text((15, 15), 'Original', font=font)
image_editable.text((1300,15), 'Jpeg', font=font)
image_editable.text((15,730), 'Neural', font=font)
image_editable.text((1300,730), 'SVD50', font=font)
img.save(os.path.join('photos', name[:-4] + '.png'))
