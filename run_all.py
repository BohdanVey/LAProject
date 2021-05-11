from svd import svd
from run_model import test_image, create_root_dir
from losses import *
import os
import cv2
import argparse
from bagoftools.namespace import Namespace
import yaml
import torch
from cae.src.models.cae_32x32x32_zero_pad_bin import CAE
from cae.src.utils import save_imgs
from random_svd import rSVD
import time
from jpeg_compression import compress_jpeg

if __name__ == '__main__':
    start_dir = 'experiments/'
    SVD_rank = 1
    svd_dir = f'svd{SVD_rank}/'
    random_svd_dir = f'random{SVD_rank}/'
    jpeg_dir = "jpeg_compression"
    neural_dir = 'neural/'
    photos_dir = 'correct/'
    result_dir = 'results/'
    total_file = f'out_all_jpeg.txt'
    filename = f'out_jpeg.txt'
    k = 0
    create_root_dir(os.path.join(start_dir, svd_dir))
    create_root_dir(os.path.join(start_dir, random_svd_dir))
    create_root_dir(os.path.join(start_dir, neural_dir))
    create_root_dir(os.path.join(start_dir, jpeg_dir))
    create_root_dir(os.path.join(result_dir))
    model = CAE()
    model.load_state_dict(torch.load('checkpoint/model_yt_small_final.state'))
    model.eval()
    model.cuda()
    print("START")
    total_svd_dice_loss = 0
    total_svd_square_loss = 0
    total_svd_time = 0

    total_rsvd_dice_loss = 0
    total_rsvd_square_loss = 0
    total_rsvd_time = 0

    total_neural_dice_loss = 0
    total_neural_square_loss = 0
    total_neural_time = 0

    total_jpeg_dice_loss = 0
    total_jpeg_square_loss = 0
    total_jpeg_time = 0
    jpeg_quality = 50
    with open(os.path.join(result_dir, total_file), 'w') as tf:
        for i in os.listdir(photos_dir):
            k += 1
            print(f"{k}/{len(os.listdir(photos_dir))}")
            img_correct = cv2.imread(os.path.join(photos_dir, i))

            # Neural Network
            st_neural = time.time()

            img_neural = test_image([os.path.join(photos_dir, i)], model)
            save_imgs(
                imgs=img_neural.unsqueeze(0),
                to_size=(3, 768, 1280),
                name=os.path.join(start_dir, neural_dir, i),
            )
            img_neural = cv2.imread(os.path.join(start_dir, neural_dir, i))
            img_neural = cv2.resize(img_neural, (1280, 720))
            neural_time = time.time() - st_neural
            total_neural_time += neural_time
            img_svd = np.zeros((720, 1280, 3))
            img_rsvd = np.zeros((720, 1280, 3))
            st_svd = time.time()
            # JPEG compression
            st_jpeg = time.time()
            jpeg_to = os.path.join(start_dir, jpeg_dir, i)[:-4] + '.jpeg'
            im_jpeg = compress_jpeg(os.path.join(photos_dir, i), jpeg_to, jpeg_quality)
            jpeg_time = time.time() - st_jpeg
            total_jpeg_time += jpeg_time
            jpeg_loss = (square_loss(im_jpeg, img_correct), dice_loss(im_jpeg, img_correct))
            total_jpeg_square_loss += jpeg_loss[0]
            total_jpeg_dice_loss += jpeg_loss[1]

            # SVD
            for q in range(3):
                values, left_s, rigth_s = svd(img_correct[:, :, q].astype('float32'), SVD_rank)

            img_svd[:, :, q] = np.matmul(np.matmul(left_s, np.diag(values)), rigth_s)
            cv2.imwrite(
                os.path.join(start_dir, svd_dir, i), img_svd
            )
            svd_time = time.time() - st_svd
            total_svd_time += svd_time

            svd_loss = (square_loss(img_svd, img_correct), dice_loss(img_svd, img_correct))
            total_svd_square_loss += svd_loss[0]
            total_svd_dice_loss += svd_loss[1]

            # Random SVD
            st_rsvd = time.time()
            for q in range(3):
                values, left_s, rigth_s = rSVD(img_correct[:, :, q].astype('float32'), SVD_rank)

            img_rsvd[:, :, q] = np.matmul(np.matmul(left_s, np.diag(values)), rigth_s)
            cv2.imwrite(
                os.path.join(start_dir, random_svd_dir, i), img_rsvd
            )
            rsvd_time = time.time() - st_rsvd
            total_rsvd_time += rsvd_time
            rsvd_loss = (square_loss(img_rsvd, img_correct), dice_loss(img_rsvd, img_correct))
            total_rsvd_square_loss += rsvd_loss[0]
            total_rsvd_dice_loss += rsvd_loss[1]

            neural_loss = (square_loss(img_neural, img_correct), dice_loss(img_neural, img_correct))
            total_neural_square_loss += neural_loss[0]
            total_neural_dice_loss += neural_loss[1]

            tf.write(f"{i}\n")

            tf.write(f"JPEG square loss: {jpeg_loss[0]}\n")
            tf.write(f"JPEG dice loss: {jpeg_loss[1]}\n")
            tf.write(f"JPEG time(sec): {jpeg_time}\n")
            with open(os.path.join(result_dir, filename), 'w') as f:
                f.write(f"JPEG square loss: {total_jpeg_square_loss / k}\n")
                f.write(f"JPEG dice loss: {total_jpeg_dice_loss / k}\n")
                f.write(f"JPEG total time(sec): {total_jpeg_time}\n")
