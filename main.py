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

def exemplar_paint(img, ksize=9):
    # Make image grey
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialise confidence values for each pixel
    # Non-empty pixels as 1
    # Empty as 0
    # ret, confidence = cv2.threshold(img_grey, 20, 1, cv2.THRESH_BINARY)
    confidence = np.zeros(img_grey.shape[:2], dtype="uint8")
    confidence[img_grey < 20] = 255
    kernel = np.ones((3,3),np.uint8)
    confidence = cv2.dilate(confidence, kernel, iterations = 1)
    confidence = 255 - confidence
    confidence = confidence.astype('float64')
    confidence /= 255
    conf3C = cv2.cvtColor(confidence.astype('uint8'), cv2.COLOR_GRAY2BGR)
    img *= conf3C
    # cv2.imshow('Confidence', confidence)

    k_area = ksize * ksize
    halfk = ksize // 2

    # Run until area is fully painted
    while True:
        # Convert image to grey
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Grey image', img_grey)

        # Threshold to get mask of paint area
        thresh = np.zeros(img_grey.shape[:2], dtype="uint8")
        thresh[img_grey < 10] = 255
        thresh = 255 - thresh

        # cv2.imshow('Binary image', thresh)

        # Check if there are still pixels to paint in
        h, w = img.shape[:2]
        sample = cv2.countNonZero(thresh)
        if sample == (h * w):
            break

        # Find the fill front of the fill area
        contour, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        contour = contour[1]

        # img2 = img.copy()
        # cv2.drawContours(image=img2, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imshow('Contours', img2)

        # Calculate the priority of each contour point
        max_priority = None
        chosen_point = None
        c_val = None
        for i in range(len(contour)):
            point = contour[i][0]
            c = 0   # Confidence
            d = 0   # Data

            # Calculate confidence for the contour point
            # Iterate through the kernal
            # Take the average of the confidence values
            confidence_patch = confidence[point[1] - halfk: point[1] + halfk + 1, point[0] - halfk: point[0] + halfk + 1]
            c = np.sum(confidence_patch) / k_area
            # for j in range(ksize):
            #     for k in range(ksize):
            #         kpoint = ((point[0]-(ksize // 2)) + j, (point[1]-(ksize // 2)) + k)
            #         if thresh[kpoint[1]][kpoint[0]] == 255:
            #             c += confidence[kpoint[1]][kpoint[0]]
            # c /= k_area

            # Calculate the normal of the contour point
            # Find the tangent
            prev_i = (i - 1) % len(contour)
            next_i = (i + 1) % len(contour)

            prev_point = contour[prev_i][0]
            next_point = contour[next_i][0]
            tan = next_point - prev_point

            # Find the vector orthogonal to the tangent
            normal = [-tan[1], tan[0]]
            normal = normal / np.linalg.norm(normal)

            # Find the isothope of the contour point
            # Calculate the image partial derivatives
            part_dir_x = cv2.Sobel(img_grey, cv2.CV_64F, 1, 0, ksize=1)
            part_dir_y = cv2.Sobel(img_grey, cv2.CV_64F, 0, 1, ksize=1)

            # empty = np.zeros(img.shape[:2], dtype="uint8")
            # part_dir = cv2.merge([empty, part_dir_x.astype('uint8'), part_dir_y.astype('uint8')])
            # cv2.imshow('Partial Derivatives', part_dir)
            # cv2.waitKey(0)

            # Find the edge orientation and magnitude of each point using the partial derivatives
            # edge_orient = math.atan(part_dir_y[point[1]][point[0]] / part_dir_x[point[1]][point[0]])
            # edge_mag = math.sqrt((part_dir_x[point[1]][point[0]] ** 2) + (part_dir_y[point[1]][point[0]] ** 2))

            # Use the edge magnitude and orientation to find the orthogonal isothope
            # isothope = [math.cos(edge_orient) * edge_mag, math.sin(edge_orient) * edge_mag]
            isothope = [-part_dir_x[point[1]][point[0]], -part_dir_y[point[1]][point[0]]]

            # copy = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)
            # cv2.line(copy, ((point[0], point[1])), ((point[0] + int(isothope[0] * 10), point[1] + int(isothope[1] * 10))), (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.line(copy, ((point[0], point[1])), ((point[0] + int(normal[0] * 10), point[1] + int(normal[1] * 10))), (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.imshow('Direction', copy)
            # cv2.waitKey(0)

            # Calculate the data value
            d = ((normal[0] * isothope[0]) + (normal[1] * isothope[1])) / 255

            priority = c * d

            if max_priority == None:
                max_priority = priority
                chosen_point = point
                c_val = c
            
            if priority > max_priority:
                max_priority = priority
                chosen_point = point
                c_val = c

        # print(chosen_point)

        # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        fill_patch = img[chosen_point[1] - halfk: chosen_point[1] + halfk + 1, chosen_point[0] - halfk: chosen_point[0] + halfk + 1]
        fill_patchmask = thresh[chosen_point[1] - halfk: chosen_point[1] + halfk + 1, chosen_point[0] - halfk: chosen_point[0] + halfk + 1] / 255
        fill_patchmask = fill_patchmask.astype('uint8')
        fill_patchmask3C = cv2.cvtColor(fill_patchmask, cv2.COLOR_GRAY2BGR)

        # fill_view_patch1 = cv2.resize(fill_patch, (225, 225), interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('fill orig', fill_view_patch1)

        min_ssd = None
        sample_patch = None
        for i in range(halfk, h - halfk + 1):
            for j in range(halfk, w - halfk + 1):
                check_patchmask = thresh[i - halfk: i + halfk + 1, j - halfk: j + halfk + 1]

                if cv2.countNonZero(check_patchmask) != k_area:
                    continue

                check_patch = img[i - halfk: i + halfk + 1, j - halfk: j + halfk + 1]

                diff = check_patch * fill_patchmask3C

                # view_patch = cv2.cvtColor(diff, cv2.COLOR_LAB2BGR)
                # view_patch = cv2.resize(diff, (225, 225), interpolation = cv2.INTER_LINEAR)
                # b, g, r = cv2.split(view_patch)
                # cv2.imshow('sample', view_patch)

                diff = fill_patch - diff
                diff = np.square(diff)
                ssd = np.sum(diff)

                # cv2.waitKey(0)

                if min_ssd == None:
                    min_ssd = ssd
                    sample_patch = check_patch
                    continue

                if ssd < min_ssd:
                    min_ssd = ssd
                    sample_patch = check_patch
        # print(min_ssd)
        # sample_patch = cv2.cvtColor(sample_patch, cv2.COLOR_LAB2BGR)
        # sample_view_patch = cv2.resize(sample_patch, (225, 225), interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('sample', sample_view_patch)
        # cv2.waitKey(0)

        fill_patchmask = 1 - fill_patchmask
        fill_patchmask3C = cv2.cvtColor(fill_patchmask, cv2.COLOR_GRAY2BGR)
        sample_patch = sample_patch * fill_patchmask3C
        fill_patch = fill_patch + sample_patch

        # fill_view_patch = cv2.resize(fill_patch, (225, 225), interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('fill', fill_view_patch)

        img[chosen_point[1] - halfk: chosen_point[1] + halfk + 1, chosen_point[0] - halfk: chosen_point[0] + halfk + 1] = fill_patch

        conf_upd = fill_patchmask.astype('float64') * c_val
        confidence[chosen_point[1] - halfk: chosen_point[1] + halfk + 1, chosen_point[0] - halfk: chosen_point[0] + halfk + 1] += conf_upd

        # cv2.waitKey(0)

    # cv2.imshow('Result', img)
    # cv2.waitKey(0)

    return img

def process(img, verbose):
    # cv2.namedWindow('collect coordinate')
    # cv2.setMouseCallback('collect coordinate', print_position)
    # cv2.imshow('collect coordinate', img)
    # while True:
    #     cv2.waitKey(1)
    #     if count == 4:
    #         break

    w, h = img.shape[:2]
    corners = np.float32([[9, 15], [233, 5], [30, 235], [249, 227]])
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
    # bilat2 = bilat.copy()
    # bilat2 = cv2.GaussianBlur(bilat, (3, 3), 2)
    # mask = cv2.dilate(mask, kernel, iterations = 2)
    # bilat[mask == 255] = bilat2[mask == 255]

    paint = exemplar_paint(bilat)

    # paint = cv2.inpaint(bilat,mask,5,cv2.INPAINT_NS)

    paint2 = paint.copy()
    paint2 = cv2.GaussianBlur(paint, (3, 3), 2)
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

    if verbose:
        cv2.imshow('orig', img)
        cv2.imshow('warp', warp)
        cv2.imshow('denoise', denoise)
        cv2.imshow('bilat', bilat)
        # cv2.imshow('median', median)
        cv2.imshow('paint', paint)
        # cv2.imshow('median2', median2)
        cv2.imshow('sharp', sharp)
        cv2.imshow('contrast', contrast)
        cv2.imshow('denoise2', denoise2)
        # cv2.imshow('bilat2', bilat2)
        # cv2.imshow('denoise3', denoise3)
        # cv2.imshow('gamma', gamma)
        # cv2.imshow('contrast2', contrast2)
        # cv2.imshow('col', col)
        # cv2.imshow('l', l)
        # cv2.imshow('s', s)
        # cv2.imshow('v', v)
        # cv2.imshow('h', h)
        cv2.waitKey(0)
    return denoise2

if __name__ == "__main__":
    path = sys.argv[1]
    if not os.path.exists("./Results"):
        os.makedirs("./Results")
    
    verbose = False
    if len(sys.argv) == 3:
        verbose = True

    for root, dirs, files in os.walk(path):
        for file in files:
            if '.jpg' not in file:
                continue
            file_path = os.path.join(root, file)
            print(file)

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            res = process(img, verbose)

            cv2.imwrite(os.path.join("./Results", file), res)
