from PIL import Image
import cv2


def compress_jpeg(src, to, quality=90):
    im1 = Image.open(src)
    im1.save(to, "JPEG", quality=quality)
    return cv2.imread(to)


if __name__ == '__main__':
    im = compress_jpeg('correct/frame_7.bmp', 'experiments/jpeg_compression/frame_7.jpeg')
