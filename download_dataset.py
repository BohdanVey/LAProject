import cv2
import os

k = 0
for i in os.listdir('yt_small_720p'):
    k += 1
    if k > 100:
        os.remove(os.path.join('yt_small_720p', i))
