import sys
import os
import cv2

if __name__ == "__main__":
    path = sys.argv[1]
    if not os.path.exists("/Results"):
        os.makedirs("./Results")

    for root, dirs, files in os.walk(path):
        for file in files:
            if '.jpg' not in file:
                continue
            file_path = os.path.join(root, file)
            print(file)

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join("./Results", file), img)
