import sys
import os
import cv2
import numpy as np
import math

count = 0
def print_position(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        count += 1
        print(f"position ({x},{y})")

def process(img):
    # cv2.namedWindow('collect coordinate')
    # cv2.setMouseCallback('collect coordinate', print_position)
    # cv2.imshow('collect coordinate', img)
    # while True:
    #     cv2.waitKey(1)
    #     if count == 4:
    #         break

    w, h = img.shape[:2]
    corners = np.float32([[8, 15], [233, 5], [30, 235], [249, 227]])
    bounds = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    M = cv2.getPerspectiveTransform(corners, bounds)
    warp = cv2.warpPerspective(img, M, (w, h))

    denoise = cv2.fastNlMeansDenoisingColored(warp, None, 13, 13, 11, 27)

    bilat = cv2.bilateralFilter(denoise,9,20,20)

    # median = cv2.medianBlur(bilat, 3)
    # median = bilat

    gray = cv2.cvtColor(bilat, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(bilat.shape[:2], dtype="uint8")
    mask[gray < 20] = 255

    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 4)

    paint = cv2.inpaint(bilat,mask,5,cv2.INPAINT_NS)

    paint2 = paint.copy()
    paint2 = cv2.medianBlur(paint, 7)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    paint[mask == 255] = paint2[mask == 255]

    # median2 = cv2.medianBlur(paint, 7)

    img_YCrCb = cv2.cvtColor(paint, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    img_YCrCb[:, :, 0] = clahe.apply(img_YCrCb[:, :, 0])
    contrast = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2BGR)

    laplace = cv2.Laplacian(contrast, cv2.CV_64F, ksize=3)

    imgf64 = np.float64(contrast)
    sharp = cv2.subtract(imgf64, laplace * 0.25)
    sharp = np.clip(sharp, 0, 255).astype('uint8')    

    denoise2 = cv2.fastNlMeansDenoisingColored(sharp, None, 3, 3, 11, 27)
    # bilat2 = cv2.bilateralFilter(denoise2,9,60,60)

    # res = cv2.medianBlur(res, 3)

    # cv2.imshow('median3', res)

    # b, g, r = cv2.split(denoise2)

    # b = np.clip(((b/255) ** 4) * 255, 0, 255).astype(np.uint8)
    # g = np.clip(((g/255) ** 2) * 255, 0, 255).astype(np.uint8)
    # col = cv2.merge([b, g, r])

    hsv = cv2.cvtColor(denoise2, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    maskb = np.zeros(denoise2.shape[:2], dtype="uint8")
    masky = np.zeros(denoise2.shape[:2], dtype="uint8")
    maskr = np.zeros(denoise2.shape[:2], dtype="uint8")
    masky[h < 50] = 255
    maskr[h < 15] = 255
    maskb[h > 100] = 255
    maskr = cv2.dilate(maskr, kernel, iterations = 1)
    # cv2.imshow('masky', masky)
    # cv2.imshow('maskr', maskr)
    # cv2.imshow('maskb', maskb)

    # s[maskr == 255] = np.clip(s[maskr == 255] * 1.1, 0, 255)
    # s[maskb == 255] = s[maskb == 255] / 1.5
    # v[maskb == 255] = v[maskb == 255] / 2
    maskw = np.zeros(denoise2.shape[:2], dtype="uint8")
    maskw[s < 50] = 255
    # cv2.imshow('maskw', maskw)
    hsv = cv2.merge([h, s, v])
    col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # l = np.clip(((l/255) ** 0.6) * 255, 0, 255).astype(np.uint8)
    # l = cv2.equalizeHist(l)
    # mask = np.zeros(col.shape[:2], dtype="uint8")
    # mask[l > 90] = 255
    # cv2.imshow('mask', mask)
    # l[maskb == 255] = l[maskb == 255] / 3

    # maskl = cv2.bitwise_and(l, l, mask=mask)
    # maskl = cv2.equalizeHist(maskl)
    # l[mask == 255] = maskl[mask == 255]

    # mask = cv2.bitwise_or(masky, maskw)
    max_val = l[maskb == 0].max()
    min_val = l[maskb == 0].min()
    target_max = 255
    target_min = 0
    l = l.astype(np.float64)
    l[maskb == 0] = (l[maskb == 0] - min_val) * ((target_max - target_min) / (max_val - min_val)) + target_min
    l = l.astype(np.uint8)

    # mask = cv2.bitwise_or(masky, maskw)
    mask = masky
    # l[mask == 0] = l[mask == 0] / 1.3
    # l[mask == 255] = np.clip(l[mask == 255] * 1.05, 0, 255)
    # l[maskr == 255] = l[maskr == 255] / 1.5

    lab = cv2.merge([l, a, b])
    gamma = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    
    # # l[maskr == 255] = np.clip((l[maskr == 255] + 10) * 1.1, 0, 255)
    # # l2 = cv2.GaussianBlur(l, (3, 3), 1)
    # # l2 = cv2.medianBlur(l2, 3)
    # # masky = cv2.dilate(masky, kernel, iterations = 2)
    # # l[masky == 255] = l2[masky == 255]
    # lab = cv2.merge([l, a, b])
    # contrast2 = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # cv2.imshow('orig', img)
    # cv2.imshow('warp', warp)
    # cv2.imshow('denoise', denoise)
    # cv2.imshow('bilat', bilat)
    # # cv2.imshow('median', median)
    # cv2.imshow('paint', paint)
    # # cv2.imshow('median2', median2)
    # cv2.imshow('sharp', sharp)
    # cv2.imshow('contrast', contrast)
    # cv2.imshow('denoise2', denoise2)
    # # cv2.imshow('bilat2', bilat2)
    # # cv2.imshow('denoise3', denoise3)
    # cv2.imshow('gamma', gamma)
    # # cv2.imshow('contrast2', contrast2)
    # cv2.imshow('col', col)
    # cv2.imshow('l', l)
    # cv2.imshow('s', s)
    # cv2.imshow('v', v)
    # cv2.imshow('h', h)
    # cv2.waitKey(0)
    return denoise2

if __name__ == "__main__":
    path = sys.argv[1]
    if not os.path.exists("./Results"):
        os.makedirs("./Results")

    for root, dirs, files in os.walk(path):
        for file in files:
            if '.jpg' not in file:
                continue
            file_path = os.path.join(root, file)
            print(file)

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            res = process(img)

            cv2.imwrite(os.path.join("./Results", file), res)
