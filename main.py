import sys
import os
import cv2
import numpy as np

count = 0
def print_position(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        count += 1
        print(f"position ({x},{y})")

def process(img):
    # cv2.imshow('orig', img)

    # cv2.namedWindow('collect coordinate')
    # cv2.setMouseCallback('collect coordinate', print_position)
    # cv2.imshow('collect coordinate', img)
    # while True:
    #     cv2.waitKey(1)
    #     if count == 4:
    #         break

    w, h = img.shape[:2]
    corners = np.float32([[8, 15], [233, 5], [29, 237], [249, 229]])
    bounds = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    M = cv2.getPerspectiveTransform(corners, bounds)
    res = cv2.warpPerspective(img, M, (w, h))

    # cv2.imshow('res', res)
    # while True:
    #     cv2.waitKey(1)
    return res

if __name__ == "__main__":
    path = sys.argv[1]
    if not os.path.exists("./Results"):
        os.makedirs("./Results")

    for root, dirs, files in os.walk(path):
        for file in files:
            if '.jpg' not in file:
                continue
            file_path = os.path.join(root, file)
            # print(file)

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            res = process(img)

            cv2.imwrite(os.path.join("./Results", file), res)
