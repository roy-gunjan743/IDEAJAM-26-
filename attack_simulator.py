import cv2

def crop(img):
    h, w = img.shape[:2]
    return img[int(h*0.2):h, int(w*0.2):w]

def compress(img):
    cv2.imwrite("temp.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return cv2.imread("temp.jpg")

def blur(img):
    return cv2.GaussianBlur(img, (7,7), 0)