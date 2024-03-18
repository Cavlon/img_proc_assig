import sys
import os
import cv2
import numpy as np

from cProfile import Profile
from pstats import SortKey, Stats

count = 0
def print_position(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        count += 1
        print(f"position ({x},{y})")

def exemplar_paint(img, ksize=9):
    # Make image grey
    # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Initialise confidence values for each pixel
    # Non-empty pixels as 1
    # Empty as 0
    thresh = np.where(img_lab[:, :, 0] < 20, 255, 0).astype('uint8')
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = (255 - thresh)
    

    # Trim image to match confidence mask
    img_lab = cv2.bitwise_and(img_lab, img_lab, mask=thresh)

    k_area = ksize * ksize
    halfk = ksize // 2

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    x, y, bw, bh = cv2.boundingRect(contours[1])
    patch = img_lab[y - halfk - 1: y + bh + halfk + 1, x - halfk - 1: x + bw + halfk + 1]
    patchthresh = thresh[y - halfk - 1: y + bh + halfk + 1, x - halfk - 1: x + bw + halfk + 1]
    patchconf = (patchthresh.astype('float64')) / 255

    # patch_view = cv2.resize(patch, (200, 200), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('Patch', patch_view)
    # patchb_view = cv2.resize(patchthresh, (200, 200), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('Patch Binary', patchb_view)

    # Calculate the image partial derivatives
    part_dir_x = cv2.Sobel(patch[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    part_dir_y = cv2.Sobel(patch[:, :, 0], cv2.CV_64F, 0, 1, ksize=3)

    # partx_view = cv2.resize(part_dir_x, (200, 200), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('PartX', partx_view)
    # party_view = cv2.resize(part_dir_y, (200, 200), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('PartY', party_view)

    h, w = img_lab.shape[:2]

    # Run until area is fully painted
    while True:
        # Convert image to grey
        # img_grey = cv2.cvtColor(img_lab, cv2.COLOR_BGR2GRAY)

        # Threshold to get mask of paint area
        # thresh = np.where(img_lab[:, :, 0] < 10, 0, 255).astype('uint8')

        # cv2.imshow('Binary image', thresh)

        # Check if there are still pixels to paint in
        # sample = cv2.countNonZero(thresh)
        # if sample == (h * w):
        #     break

        # Find the fill front of the fill area
        contours, hierarchy = cv2.findContours(image=patchthresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        if len(contours) == 1:
            break

        # contour = contours[1]

        # conts = patch.copy()
        # cv2.drawContours(image=conts, contours=contours[1:], contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        # conts_view = cv2.resize(conts, (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Contours', conts_view)

        # Calculate the priority of each contour point
        max_priority = None
        chosen_point = None
        c_val = None
        for i in range(1, len(contours)):
            contour = contours[i]
            for j in range(len(contour)):
                point = contour[j][0]
                c = 0   # Confidence
                d = 0   # Data

                # Calculate confidence for the contour point
                # Take a patch of conifidence values around the point
                # Take the average of the confidence values
                confidence_patch = patchconf[point[1] - halfk: point[1] + halfk + 1, point[0] - halfk: point[0] + halfk + 1]
                c = np.sum(confidence_patch) / k_area

                # Calculate the normal of the contour point
                # Find the tangent
                prev_i = (j - 1) % len(contour)
                next_i = (j + 1) % len(contour)

                prev_point = contour[prev_i][0]
                next_point = contour[next_i][0]
                tan = next_point - prev_point

                # Find the vector orthogonal to the tangent
                normal = [-tan[1], tan[0]]
                normal = normal / np.linalg.norm(normal)

                x_start, x_end = point[0] - halfk, point[0] + halfk + 1
                y_start, y_end = point[1] - halfk, point[1] + halfk + 1

                part_dir_patch_x = part_dir_x[y_start:y_end, x_start:x_end]
                part_dir_patch_y = part_dir_y[y_start:y_end, x_start:x_end]

                Ixx = np.square(part_dir_patch_x)
                Ixy = part_dir_patch_x * part_dir_patch_y
                Iyy = np.square(part_dir_patch_y)

                Sxx = np.sum(Ixx)
                Sxy = np.sum(Ixy)
                Syy = np.sum(Iyy)

                S = np.array([[Sxx, Sxy], [Sxy, Syy]])

                eigenvalues, eigenvectors = np.linalg.eig(S)
                # print(eigenvectors)
                # print(eigenvalues)
                max_eigenvalue_ind = np.argmax(eigenvalues)
                max_eigenvector = eigenvectors[:, max_eigenvalue_ind]
                # print(max_eigenvector)
                # direction = patch.copy()
                # cv2.line(direction, ((point[0], point[1])), ((point[0] + int(max_eigenvector[0]), point[1] + int(max_eigenvector[1]))), (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.line(direction, ((point[0], point[1])), ((point[0] + int(normal[0] * 10), point[1] + int(normal[1] * 10))), (0, 0, 255), 2, cv2.LINE_AA)
                # direction_view = cv2.resize(direction, (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('Direction', direction_view)
                # cv2.waitKey(0)

                d = abs(np.dot(max_eigenvector, normal)) * (eigenvalues[max_eigenvalue_ind] / 255)

                # part_dir_patch = np.square(part_dir_x[y_start:y_end, x_start:x_end]) + np.square(part_dir_y[y_start:y_end, x_start:x_end])
                # isothope_point = np.unravel_index(part_dir_patch.argmax(), part_dir_patch.shape)

                # Find the isothope of the contour point
                # Use the image partial derivatives to find the isothope of the point
                # isothope = [-part_dir_x[y_start:y_end, x_start:x_end][isothope_point], -part_dir_y[y_start:y_end, x_start:x_end][isothope_point]]

                # direction = patch.copy()
                # cv2.line(direction, ((point[0], point[1])), ((point[0] + int(isothope[0] * 10), point[1] + int(isothope[1] * 10))), (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.line(direction, ((point[0], point[1])), ((point[0] + int(normal[0] * 10), point[1] + int(normal[1] * 10))), (0, 0, 255), 2, cv2.LINE_AA)
                # direction_view = cv2.resize(direction, (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('Direction', direction_view)

                # Calculate the data value
                # d = ((normal[0] * isothope[0]) + (normal[1] * isothope[1])) / 255
                # d = 1

                # Use the confidence and data values to calculate the point's priority
                priority = c * d

                # Find the point with highest priority
                if max_priority == None:
                    max_priority = priority
                    chosen_point = point
                    c_val = c
                
                if priority > max_priority:
                    max_priority = priority
                    chosen_point = point
                    c_val = c
        k_inds = (chosen_point[1] - halfk, chosen_point[1] + halfk + 1, chosen_point[0] - halfk, chosen_point[0] + halfk + 1)

        # Get the corresponding patch for the chosen point in the image
        fill_patch = patch[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]]
        fill_patchmask = patchthresh[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] / 255
        fill_patchmask = fill_patchmask.astype('uint8')

        # fill_view_patch1 = cv2.resize(fill_patch, (225, 225), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('fill orig', fill_view_patch1)

        # Find the most similar patch in the rest of the image
        min_ssd = None
        sample_patch = None

        # Iterate through every possible patch
        for i in range(halfk, h//5 - halfk):
            for j in range(w//2 + halfk, w - halfk):
                check_patchmask = thresh[i - halfk: i + halfk + 1, j - halfk: j + halfk + 1]

                # Make sure the patch doesn't overlap the empty area
                if np.any(check_patchmask == 0):
                    continue

                check_patch = img_lab[i - halfk: i + halfk + 1, j - halfk: j + halfk + 1]
                # check_view_patch = cv2.resize(check_patch, (225, 225), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('check', check_view_patch)
                # cv2.waitKey(0)

                # Find the ssd of the non-empty indices
                # if check_patch.shape[:2] != fill_patchmask.shape:
                #     print()
                #     print(i)
                #     print(j)
                #     print(check_patch.shape[0])
                #     print(check_patch.shape[1])
                #     check_view_patch = cv2.resize(check_patch, (225, 225), interpolation = cv2.INTER_NEAREST)
                #     cv2.imshow('check', check_view_patch)
                #     cv2.waitKey(0)
                diff = cv2.bitwise_and(check_patch, check_patch, mask=fill_patchmask)
                diff = fill_patch - diff
                diff = np.square(diff)
                ssd = np.sum(diff)

                # Find the patch that minimises the ssd (distance)
                if min_ssd == None:
                    min_ssd = ssd
                    sample_patch = check_patch
                    continue

                if ssd < min_ssd:
                    min_ssd = ssd
                    sample_patch = check_patch
        # sample_view_patch = cv2.resize(sample_patch, (225, 225), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('sample', sample_view_patch)

        fill_patchmask = 1 - fill_patchmask
        sample_patch = cv2.bitwise_and(sample_patch, sample_patch, mask=fill_patchmask)
        # fill_patch = fill_patch + sample_patch

        # fill_view_patch = cv2.resize(fill_patch, (225, 225), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('fill', fill_view_patch)

        # Replace the empty pixels with the values from the sample
        patch[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] += sample_patch
        # fill_view_patch = cv2.resize(patch[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]], (225, 225), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('fill', fill_view_patch)

        # Update the confidence values
        conf_upd = fill_patchmask.astype('float64') * c_val
        patchconf[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] += conf_upd
        patchthresh[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] = 255

        thresh[y - halfk - 1: y + bh + halfk + 1, x - halfk - 1: x + bw + halfk + 1] = patchthresh

        patch_part_dir_x = cv2.Sobel(patch[k_inds[0] - 1: k_inds[1] + 1, k_inds[2] - 1: k_inds[3] + 1, 0], cv2.CV_64F, 1, 0, ksize=3)
        patch_part_dir_y = cv2.Sobel(patch[k_inds[0] - 1: k_inds[1] + 1, k_inds[2] - 1: k_inds[3] + 1, 0], cv2.CV_64F, 0, 1, ksize=3)

        part_dir_x[k_inds[0] - 1: k_inds[1] + 1, k_inds[2] - 1: k_inds[3] + 1] = patch_part_dir_x
        part_dir_y[k_inds[0] - 1: k_inds[1] + 1, k_inds[2] - 1: k_inds[3] + 1] = patch_part_dir_y

        # patch_view = cv2.resize(patch, (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Patch', patch_view)
        # patchb_view = cv2.resize(patchthresh, (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Patch Binary', patchb_view)
        # patchconf_view = cv2.resize((patchconf * 255).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Patch Conf', patchconf_view)
        # partx_view = cv2.resize(part_dir_x, (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('PartX', partx_view)
        # party_view = cv2.resize(part_dir_y, (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('PartY', party_view)
        # cv2.waitKey(0)
    img_lab[y - halfk - 1: y + bh + halfk + 1, x - halfk - 1: x + bw + halfk + 1] = patch
    res = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    # cv2.imshow('Result', res)
    # cv2.waitKey(0)
    return res

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

    paint = exemplar_paint(bilat, 13)

    # paint = cv2.inpaint(bilat,mask,5,cv2.INPAINT_NS)

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

    # hsv = cv2.cvtColor(denoise2, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)

    # maskb = np.zeros(denoise2.shape[:2], dtype="uint8")
    # masky = np.zeros(denoise2.shape[:2], dtype="uint8")
    # maskr = np.zeros(denoise2.shape[:2], dtype="uint8")
    # masky[h < 50] = 255
    # maskr[h < 15] = 255
    # maskb[h > 100] = 255
    # maskr = cv2.dilate(maskr, kernel, iterations = 1)
    # # cv2.imshow('masky', masky)
    # # cv2.imshow('maskr', maskr)
    # # cv2.imshow('maskb', maskb)

    # # s[maskr == 255] = np.clip(s[maskr == 255] * 1.1, 0, 255)
    # # s[maskb == 255] = s[maskb == 255] / 1.5
    # # v[maskb == 255] = v[maskb == 255] / 2
    # maskw = np.zeros(denoise2.shape[:2], dtype="uint8")
    # maskw[s < 50] = 255
    # # cv2.imshow('maskw', maskw)
    # hsv = cv2.merge([h, s, v])
    # col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # # l = np.clip(((l/255) ** 0.6) * 255, 0, 255).astype(np.uint8)
    # # l = cv2.equalizeHist(l)
    # # mask = np.zeros(col.shape[:2], dtype="uint8")
    # # mask[l > 90] = 255
    # # cv2.imshow('mask', mask)
    # # l[maskb == 255] = l[maskb == 255] / 3

    # # maskl = cv2.bitwise_and(l, l, mask=mask)
    # # maskl = cv2.equalizeHist(maskl)
    # # l[mask == 255] = maskl[mask == 255]

    # # mask = cv2.bitwise_or(masky, maskw)
    # max_val = l[maskb == 0].max()
    # min_val = l[maskb == 0].min()
    # target_max = 255
    # target_min = 0
    # l = l.astype(np.float64)
    # l[maskb == 0] = (l[maskb == 0] - min_val) * ((target_max - target_min) / (max_val - min_val)) + target_min
    # l = l.astype(np.uint8)

    # # mask = cv2.bitwise_or(masky, maskw)
    # mask = masky
    # # l[mask == 0] = l[mask == 0] / 1.3
    # # l[mask == 255] = np.clip(l[mask == 255] * 1.05, 0, 255)
    # # l[maskr == 255] = l[maskr == 255] / 1.5

    # lab = cv2.merge([l, a, b])
    # gamma = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

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
