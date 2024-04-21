import sys
import os
import cv2
import numpy as np

def exemplar_paint(img, ksize=9):
    h, w = img.shape[:2]

    # Use the CIELAB colour space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img_val = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    # cv2.imshow('grey', img_val)
    # cv2.imshow('lab', img_lab)

    # Create a mask for the area to fill in
    thresh = np.where(img_val < 20, 0, 255).astype('uint8')
    # cv2.imshow('thresh', thresh)

    halfk = ksize // 2

    # Isolate a small area around the fill area to work on
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # conts = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(image=conts, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    # conts_view = cv2.resize(conts, (200, 200), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('Contours', conts_view)
    x = None
    y = None
    bw = None
    bh = None
    for i in range(1, len(contours)):
        x, y, bw, bh = cv2.boundingRect(contours[i])
        if (bw > ksize * 2) and (bh > ksize * 2):
            thresh = np.zeros(img.shape[:2], dtype="uint8")
            cv2.drawContours(thresh, [contours[i]], 0, color=255, thickness=-1)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            thresh = 255 - thresh
            break
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    x, y, bw, bh = cv2.boundingRect(contours[1])
    # Trim image to match the mask
    img_lab = cv2.bitwise_and(img_lab, img_lab, mask=thresh)
    patch = img_lab[y - halfk - 1: y + bh + halfk + 1, x - halfk - 1: x + bw + halfk + 1]
    patchthresh = thresh[y - halfk - 1: y + bh + halfk + 1, x - halfk - 1: x + bw + halfk + 1]
    patchconf = (patchthresh.astype('float64')) / 255
    patchgrey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('thresh', thresh)
    # patch_view = cv2.resize(patch, (200, 200), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('Patch', patch_view)
    # patchb_view = cv2.resize(patchthresh, (200, 200), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('Patch Binary', patchb_view)
    # cv2.waitKey(0)

    # Calculate the patch partial derivatives using luminosity
    part_dir_x = cv2.Sobel(patchgrey, cv2.CV_64F, 1, 0, ksize=3)
    part_dir_y = cv2.Sobel(patchgrey, cv2.CV_64F, 0, 1, ksize=3)
    mask = 255 - patchthresh
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = 255 - mask
    part_dir_x = cv2.bitwise_and(part_dir_x, part_dir_x, mask=mask)
    part_dir_y = cv2.bitwise_and(part_dir_y, part_dir_y, mask=mask)

    # Construct a strcuture tensor for the patch
    Ixx = np.square(part_dir_x)
    Ixy = part_dir_x * part_dir_y
    Iyy = np.square(part_dir_x)

    # partx_view = cv2.resize(((part_dir_x * (255/(np.max(part_dir_x) - np.min(part_dir_x)))) - np.min(part_dir_x * (255/(np.max(part_dir_x) - np.min(part_dir_x))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('PartX', partx_view)
    # party_view = cv2.resize(((part_dir_y * (255/(np.max(part_dir_y) - np.min(part_dir_y)))) - np.min(part_dir_y * (255/(np.max(part_dir_y) - np.min(part_dir_y))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('PartY', party_view)

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
            cont_length = len(contour)
            for j in range(cont_length):
                point = contour[j][0]
                c = 0   # Confidence
                d = 0   # Data

                x_start, x_end = point[0] - halfk, point[0] + halfk + 1
                y_start, y_end = point[1] - halfk, point[1] + halfk + 1

                # Calculate confidence for the contour point
                # Take a patch of confidence values around the point
                # Take the average of the confidence values
                confidence_patch = patchconf[y_start: y_end, x_start: x_end]
                c = np.mean(confidence_patch)
                c = c ** 2

                # labshow = cv2.resize(patch[point[1] - halfk: point[1] + halfk + 1, point[0] - halfk: point[0] + halfk + 1], (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('labpatch', labshow)
                # lshow = cv2.resize(patchgrey[point[1] - halfk: point[1] + halfk + 1, point[0] - halfk: point[0] + halfk + 1], (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('lpatch', lshow)

                # Calculate the normal of the contour point
                # Find the tangent
                prev_i = (j - 1) % cont_length
                next_i = (j + 1) % cont_length

                prev_point = contour[prev_i][0]
                next_point = contour[next_i][0]
                tan = next_point - prev_point

                # Find the vector orthogonal to the tangent
                normal = [-tan[1], tan[0]]
                normal = normal / np.linalg.norm(normal)

                # Calculate the orthogonal isothope
                # Isolate the local partial derivatives
                # part_dir_patch_x = part_dir_x[y_start:y_end, x_start:x_end]
                # part_dir_patch_y = part_dir_y[y_start:y_end, x_start:x_end]
                # dir_mask = mask[y_start:y_end, x_start:x_end]
                # mask = 255 - mask
                # kernel = np.ones((3,3),np.uint8)
                # mask = cv2.dilate(mask, kernel, iterations = 1)
                # mask = 255 - mask

                Ixx_patch = Ixx[y_start:y_end, x_start:x_end]
                Ixy_patch = Ixy[y_start:y_end, x_start:x_end]
                Iyy_patch = Iyy[y_start:y_end, x_start:x_end]

                # mask_view = cv2.resize(patchthresh[point[1] - halfk: point[1] + halfk + 1, point[0] - halfk: point[0] + halfk + 1], (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('mask', mask_view)
                # a = Ixx_patch.copy()
                # Ixx_view = cv2.resize(((a * (255/(np.max(a) - np.min(a)))) - np.min(a * (255/(np.max(a) - np.min(a))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('Ixx', Ixx_view)
                # b = Iyy_patch.copy()
                # Iyy_view = cv2.resize(((b * (255/(np.max(b) - np.min(b)))) - np.min(b * (255/(np.max(b) - np.min(b))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('Iyy', Iyy_view)
                # e = Ixy_patch.copy()
                # Ixy_view = cv2.resize(((e * (255/(np.max(e) - np.min(e)))) - np.min(e * (255/(np.max(e) - np.min(e))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('Ixy', Ixy_view)
                # partx_view = cv2.resize(((part_dir_patch_x * (255/(np.max(part_dir_patch_x) - np.min(part_dir_patch_x)))) - np.min(part_dir_patch_x * (255/(np.max(part_dir_patch_x) - np.min(part_dir_patch_x))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('PartX', partx_view)
                # party_view = cv2.resize(((part_dir_patch_y * (255/(np.max(part_dir_patch_y) - np.min(part_dir_patch_y)))) - np.min(part_dir_patch_y * (255/(np.max(part_dir_patch_y) - np.min(part_dir_patch_y))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('PartY', party_view)

                # cv2.imshow('Ixx', Ixx)
                # cv2.imshow('Ixx Patch', Ixx[mask[y_start: y_end, x_start: x_end] == 255])

                Sxx = np.mean(Ixx_patch[mask[y_start: y_end, x_start: x_end] == 255])
                Sxy = np.mean(Ixy_patch[mask[y_start: y_end, x_start: x_end] == 255])
                Syy = np.mean(Iyy_patch[mask[y_start: y_end, x_start: x_end] == 255])

                # if Sxx == np.nan:
                #     print("NAN")
                #     Sxx = 0
                #     Sxy = 0
                #     Syy = 0
                    
                # print(np.sum(Ixx_patch[mask[y_start: y_end, x_start: x_end] == 255]))
                # print(len(Ixx_patch[mask[y_start: y_end, x_start: x_end] == 255]))
                # patch_view = cv2.resize(patch[y_start: y_end, x_start: x_end], (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('Patch', patch_view)
                # patchb_view = cv2.resize(patchthresh[y_start: y_end, x_start: x_end], (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('Patch Binary', patchb_view)
                
                # print(Ixx_patch)
                # print(mask[y_start: y_end, x_start: x_end])
                # print(Sxx)
                # print(type(Sxx))
                # cv2.waitKey(0)

                S = np.array([[Sxx, Sxy], [Sxy, Syy]])

                # Calculate the structure tensor's eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(S)

                # Find the strongest eigenvector (orthogonal isothope)
                if eigenvalues[0] > eigenvalues[1]:
                    max_eigenvalue_ind = 0
                else:
                    max_eigenvalue_ind = 1
                max_eigenvector = eigenvectors[:, max_eigenvalue_ind]
                max_eigenvector = np.array([-max_eigenvector[1], max_eigenvector[0]])
                max_eigenvector *= eigenvalues[max_eigenvalue_ind]
                # print(normal)
                # print(max_eigenvector)
                # print(abs(np.dot(max_eigenvector, normal)))
                # print(eigenvalues[max_eigenvalue_ind])

                # direction = patch.copy()
                # cv2.line(direction, ((point[0], point[1])), ((point[0] + int(max_eigenvector[0]), point[1] + int(max_eigenvector[1]))), (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.line(direction, ((point[0], point[1])), ((point[0] + int(normal[0] * 10), point[1] + int(normal[1] * 10))), (0, 0, 255), 1, cv2.LINE_AA)
                # direction_view = cv2.resize(direction, (200, 200), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow('Direction', direction_view)
                # cv2.waitKey(0)

                # Calculate the data value
                d = abs(np.dot(max_eigenvector, normal)) / 255

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
                    # eig = max_eigenvector
                
                if priority > max_priority:
                    max_priority = priority
                    chosen_point = point
                    c_val = c
                    # eig = max_eigenvector
        k_inds = (chosen_point[1] - halfk, chosen_point[1] + halfk + 1, chosen_point[0] - halfk, chosen_point[0] + halfk + 1)
        # direction = patch.copy()
        # cv2.line(direction, ((chosen_point[0], chosen_point[1])), ((chosen_point[0] + int(eig[0]), point[1] + int(eig[1]))), (0, 255, 0), 1, cv2.LINE_AA)
        # direction_view = cv2.resize(direction, (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Direction', direction_view)
        # cv2.waitKey(0)

        # Get the corresponding patch for the chosen point in the image
        fill_patch = patch[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]]
        fill_patchmask = patchthresh[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] / 255
        fill_patchmask = fill_patchmask.astype('uint8')

        # fill_view_patch1 = cv2.resize(fill_patch, (225, 225), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('fill orig', fill_view_patch1)

        # Find the most similar patch in the rest of the image
        min_ssd = None
        sample_patch = None

        kernel = np.ones((ksize+1, ksize+1), np.uint8)
        overlap_check = cv2.dilate(255 - thresh, kernel, iterations=1)
        # cv2.imshow('overlap', overlap_check)

        # Iterate through every possible patch in the source region
        for i in range(halfk, h//5 - halfk, 2):
            for j in range(w//2 + halfk, w - halfk, 2):

                # Make sure the patch doesn't overlap the empty area
                if overlap_check[i, j] == 255:
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

                # Calculate the SSD (similarity) of the patches
                diff = cv2.bitwise_and(check_patch, check_patch, mask=fill_patchmask)
                diff = fill_patch - diff
                diff = np.square(diff)
                ssd = np.sum(diff)

                # Find the patch that minimises the SSD (distance)
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
        sample_patch_grey = cv2.cvtColor(sample_patch, cv2.COLOR_BGR2GRAY)
        patchgrey[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] += sample_patch_grey
        # fill_view_patch = cv2.resize(patch[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]], (225, 225), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('fill', fill_view_patch)

        # Update the confidence values and mask
        conf_upd = fill_patchmask.astype('float64') * c_val
        patchconf[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] += conf_upd
        patchthresh[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] = 255

        thresh[y - halfk - 1: y + bh + halfk + 1, x - halfk - 1: x + bw + halfk + 1] = patchthresh

        # Update the partial derivatives
        patch_part_dir_x = cv2.Sobel(patchgrey[k_inds[0] - 1: k_inds[1] + 1, k_inds[2] - 1: k_inds[3] + 1], cv2.CV_64F, 1, 0, ksize=3)
        patch_part_dir_y = cv2.Sobel(patchgrey[k_inds[0] - 1: k_inds[1] + 1, k_inds[2] - 1: k_inds[3] + 1], cv2.CV_64F, 0, 1, ksize=3)
        patch_part_dir_x = patch_part_dir_x[1:-1, 1:-1]
        patch_part_dir_y = patch_part_dir_y[1:-1, 1:-1]

        patch_Ixx = patch_part_dir_x ** 2
        patch_Ixy = patch_part_dir_x * patch_part_dir_y
        patch_Iyy = patch_part_dir_y ** 2

        Ixx[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] = patch_Ixx
        Ixy[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] = patch_Ixy
        Iyy[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] = patch_Iyy

        mask = 255 - patchthresh
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)
        mask = 255 - mask
        Ixx = cv2.bitwise_and(Ixx, Ixx, mask=mask)
        Ixy = cv2.bitwise_and(Ixy, Ixy, mask=mask)
        Iyy = cv2.bitwise_and(Iyy, Iyy, mask=mask)

        # a = cv2.resize(((part_dir_x * (255/(np.max(part_dir_x) - np.min(part_dir_x)))) - np.min(part_dir_x[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] * (255/(np.max(part_dir_x[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]]) - np.min(part_dir_x[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]]))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('a', a)
        # b = cv2.resize(patch_part_dir_x[1:-1, 1:-1], (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('b', b)

        # patch_view = cv2.resize(patch, (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Patch', patch_view)
        # patchb_view = cv2.resize(patchthresh, (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Patch Binary', patchb_view)
        # patchconf_view = cv2.resize((patchconf * 255).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Patch Conf', patchconf_view)
        # Ixx_view = cv2.resize(((Ixx * (255/(np.max(Ixx) - np.min(Ixx)))) - np.min(Ixx * (255/(np.max(Ixx) - np.min(Ixx))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Ixx', Ixx_view)
        # Ixy_view = cv2.resize(((Ixy * (255/(np.max(Ixy) - np.min(Ixy)))) - np.min(Ixy * (255/(np.max(Ixy) - np.min(Ixy))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Ixy', Ixy_view)
        # Iyy_view = cv2.resize(((Iyy * (255/(np.max(Iyy) - np.min(Iyy)))) - np.min(Iyy * (255/(np.max(Iyy) - np.min(Iyy))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('Iyy', Iyy_view)
        # partx_view = cv2.resize(((patch_part_dir_x * (255/(np.max(patch_part_dir_x) - np.min(patch_part_dir_x)))) - np.min(patch_part_dir_x * (255/(np.max(patch_part_dir_x) - np.min(patch_part_dir_x))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('PartX', partx_view)
        # party_view = cv2.resize(((patch_part_dir_y * (255/(np.max(patch_part_dir_y) - np.min(patch_part_dir_y)))) - np.min(patch_part_dir_y * (255/(np.max(patch_part_dir_y) - np.min(patch_part_dir_y))))).astype('uint8'), (200, 200), interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('PartY', party_view)
        # cv2.waitKey(0)
    img_lab[y - halfk - 1: y + bh + halfk + 1, x - halfk - 1: x + bw + halfk + 1] = patch
    res = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    # cv2.imshow('Result', res)
    # cv2.waitKey(0)
    return res

def kuwahara_filter_color(img, ksize, sigma):
    w, h = img.shape[:2]
    halfk = ksize//2
    # cv2.imshow("orig", img)

    padded = cv2.copyMakeBorder(img, halfk, halfk, halfk, halfk, cv2.BORDER_REFLECT)

    img_lab = cv2.cvtColor(padded, cv2.COLOR_BGR2Lab).astype(np.float64)

    filtered_image = np.zeros(img.shape, dtype="float64")

    squared_l = img_lab[:, :, 0] ** 2

    # kernel = np.ones((halfk+1,halfk+1),np.float32)/((halfk+1)**2)
    # mean_l = cv2.filter2D(l,-1,kernel)
    # std_l = cv2.filter2D(squared_l,-1,kernel) - (mean_l ** 2)
    # mean = cv2.filter2D(img_lab, -1, kernel)

    kernel = cv2.getGaussianKernel(ksize, sigma)
    zs = np.zeros((halfk, 1))
    kleft = kernel[:halfk+1]
    kright = kernel[halfk:]
    ker_norm = np.sum(kleft)
    kleft = kleft / ker_norm
    kright = kright / ker_norm
    kleft = np.concatenate((kleft, zs))
    kright = np.concatenate((zs, kright))
    
    segments = np.array([cv2.sepFilter2D(img_lab, -1, kleft, kleft), cv2.sepFilter2D(img_lab, -1, kleft, kright), cv2.sepFilter2D(img_lab, -1, kright, kleft), cv2.sepFilter2D(img_lab, -1, kright, kright)])
    mean_l = np.array([segments[0][:,:,0], segments[1][:,:,0], segments[2][:,:,0], segments[3][:,:,0]])
    std_l = np.array([np.sqrt(np.abs(cv2.sepFilter2D(squared_l, -1, kleft, kleft) - (mean_l[0] ** 2))), np.sqrt(np.abs(cv2.sepFilter2D(squared_l, -1, kleft, kright) - (mean_l[1] ** 2))), np.sqrt(np.abs(cv2.sepFilter2D(squared_l, -1, kright, kleft) - (mean_l[2] ** 2))), np.sqrt(np.abs(cv2.sepFilter2D(squared_l, -1, kright, kright) - (mean_l[3] ** 2)))])
    weights = np.array([1 / (1 + std_l[0]), 1 / (1 + std_l[1]), 1 / (1 + std_l[2]), 1 / (1 + std_l[3])])

    # mean_l = cv2.GaussianBlur(l, (halfk+1, halfk+1), sigma)
    # std_l = cv2.GaussianBlur(squared_l, (halfk+1, halfk+1), sigma) - (mean_l ** 2)
    # mean = cv2.GaussianBlur(img_lab, (halfk+1, halfk+1), sigma)
    # weights = 1 / (1 + std_l)

    # quad_pos = halfk // 2
    
    for y in range(halfk, h + halfk):
        for x in range(halfk, w + halfk):
            # ws = np.array([weights[y-quad_pos][x-quad_pos], weights[y+quad_pos][x-quad_pos], weights[y-quad_pos][x+quad_pos], weights[y+quad_pos][x+quad_pos]])
            # means = np.array([mean[y-quad_pos][x-quad_pos], mean[y+quad_pos][x-quad_pos], mean[y-quad_pos][x+quad_pos], mean[y+quad_pos][x+quad_pos]])
            # print(means)
            # print(means.transpose())
            # print(ws)
            # print(means.transpose() * ws)

            # print(segments[0][y, x])
            # print(weights[0][y, x])

            wsum = 0
            for i in range(4):
                wsum += weights[i][y, x]
                filtered_image[y-halfk][x-halfk] += segments[i][y, x] * weights[i][y, x]
            filtered_image[y-halfk][x-halfk] /= wsum

    filtered_image = filtered_image.astype(np.uint8)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_Lab2BGR)
    # cv2.imshow("kuwa", filtered_image)
    # cv2.waitKey(0)
    return filtered_image

def process(img, verbose, test=False):
    w, h = img.shape[:2]
    corners = np.float32([[9, 15], [233, 5], [30, 235], [249, 227]])
    bounds = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    M = cv2.getPerspectiveTransform(corners, bounds)
    warp = cv2.warpPerspective(img, M, (w, h))

    # kuwahara_filter_color(warp)

    # median = cv2.medianBlur(warp, 3)
    median = kuwahara_filter_color(warp, 5, 1.5)
    # cv2.imshow("a", median)
    # cv2.waitKey(0)

    denoise = cv2.fastNlMeansDenoisingColored(median, None, 10, 10, 11, 27)

    bilat = cv2.bilateralFilter(denoise,9,20,20)

    img_YCrCb = cv2.cvtColor(bilat, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(10, 10))
    img_YCrCb[:, :, 0] = clahe.apply(img_YCrCb[:, :, 0])
    contrast = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2BGR)

    paint = exemplar_paint(contrast, 9)

    gray = cv2.cvtColor(bilat, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(bilat.shape[:2], dtype="uint8")
    mask[gray < 20] = 255

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 3)

    paint2 = paint.copy()
    paint2 = cv2.medianBlur(paint2, 7)
    paint2 = cv2.GaussianBlur(paint2, (3, 3), 2)
    paint[mask == 255] = paint2[mask == 255]

    kuwa = kuwahara_filter_color(paint, 5, 0.75)

    laplace = cv2.Laplacian(kuwa, cv2.CV_64F, ksize=3)

    imgf64 = np.float64(kuwa)
    sharp = cv2.subtract(imgf64, laplace * 1.1)
    sharp = np.clip(sharp, 0, 255).astype('uint8')    

    # denoise2 = cv2.fastNlMeansDenoisingColored(sharp, None, 7, 7, 11, 15)
    denoise2 = cv2.bilateralFilter(sharp,9,40,40)
    # denoise2 = cv2.medianBlur(sharp, 3)

    # img_lab = cv2.cvtColor(denoise2, cv2.COLOR_BGR2Lab)
    # l = img_lab[:,:,0] / 255
    # l = np.power(l, 1.15) * 0.95
    # img_lab[:,:,0] = np.clip(l * 255, 0, 255)
    # # np.clip(img_lab[:, :, 0], 0, 255)
    # gamma = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    if test:
        cv2.imwrite(os.path.join("./test_imgs", "orig.jpg"), img)
        cv2.imwrite(os.path.join("./test_imgs", "warp.jpg"), warp)
        cv2.imwrite(os.path.join("./test_imgs", "median.jpg"), median)
        cv2.imwrite(os.path.join("./test_imgs", "denoise.jpg"), denoise)
        cv2.imwrite(os.path.join("./test_imgs", "bilat.jpg"), bilat)
        cv2.imwrite(os.path.join("./test_imgs", "contrast.jpg"), contrast)
        cv2.imwrite(os.path.join("./test_imgs", "paint.jpg"), paint)
        cv2.imwrite(os.path.join("./test_imgs", "sharp.jpg"), sharp)
        cv2.imwrite(os.path.join("./test_imgs", "denoise2.jpg"), denoise2)
    if verbose:
        cv2.imshow('orig', img)
        cv2.imshow('warp', warp)
        cv2.imshow('denoise', denoise)
        cv2.imshow('bilat', bilat)
        cv2.imshow('median', median)
        cv2.imshow('paint', paint)
        cv2.imshow('sharp', sharp)
        cv2.imshow('contrast', contrast)
        cv2.imshow('denoise2', denoise2)
        cv2.imshow('gamma', gamma)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return denoise2

if __name__ == "__main__":
    path = sys.argv[1]
    if not os.path.exists("./Results"):
        os.makedirs("./Results")
    
    select = None
    verbose = False
    if len(sys.argv) == 3:
        verbose = True
        if sys.argv[2].isdigit():
            select = sys.argv[2]

    for root, dirs, files in os.walk(path):
        for file in files:
            if '.jpg' not in file:
                continue
            if select:
                if not (select in file):
                    continue
                file_path = os.path.join(root, file)
                print(file)

                img = cv2.imread(file_path, cv2.IMREAD_COLOR)

                res = process(img, verbose)
            file_path = os.path.join(root, file)
            print(file)

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            res = process(img, verbose)

            cv2.imwrite(os.path.join("./Results", file), res)
