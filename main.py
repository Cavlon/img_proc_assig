import sys
import os
import cv2
import numpy as np


def exemplar_paint(img, ksize=9):
    h, w = img.shape[:2]

    # Use the CIELAB colour space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Use the value for thresholding
    img_val = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    # Create a mask for the area to fill in
    thresh = np.where(img_val < 20, 0, 255).astype('uint8')

    halfk = ksize // 2

    # Isolate a small area around the fill area to work on
    contours, hierarchy = cv2.findContours(image=thresh,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    # Isolate the correct missing region
    x = None
    y = None
    bw = None
    bh = None

    # There may be several contours the thresholding returns
    # The large region in the top left is the correct one to be isolated
    for i in range(1, len(contours)):
        # Calculate the bounding box for the contour
        x, y, bw, bh = cv2.boundingRect(contours[i])

        # If the box is large then it is the missing region's bounding box
        # Ammend the thresholded image to only have the correct missing region
        if (bw > ksize * 2) and (bh > ksize * 2):
            thresh = np.zeros(img.shape[:2], dtype="uint8")
            cv2.drawContours(thresh, [contours[i]], 0, color=255, thickness=-1)
            kernel = np.ones((3, 3), np.uint8)

            # Expand the fill region slightly to improve boundary smoothness
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            thresh = 255 - thresh
            break

    # Find the contour of the ammended thresholded image
    contours, hierarchy = cv2.findContours(image=thresh,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    # Calculate the bounding box for the missing region
    x, y, bw, bh = cv2.boundingRect(contours[1])

    # Isolate a small area around the fill area to process
    img_lab = cv2.bitwise_and(img_lab, img_lab, mask=thresh)
    patch = img_lab[y - halfk - 1: y + bh + halfk + 1,
                    x - halfk - 1: x + bw + halfk + 1]

    # Thresholded patch as a fill area mask
    patchthresh = thresh[y - halfk - 1: y + bh + halfk + 1,
                         x - halfk - 1: x + bw + halfk + 1]

    # Patch confidence values
    patchconf = (patchthresh.astype('float64')) / 255

    # Greyscale patch for partial derivative calculation
    patchgrey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Calculate the patch partial derivatives
    part_dir_x = cv2.Sobel(patchgrey, cv2.CV_64F, 1, 0, ksize=3)
    part_dir_y = cv2.Sobel(patchgrey, cv2.CV_64F, 0, 1, ksize=3)

    # Expand the mask
    mask = 255 - patchthresh
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = 255 - mask

    # Mask the partial derivatives so the missing region doesn't affect them
    part_dir_x = cv2.bitwise_and(part_dir_x, part_dir_x, mask=mask)
    part_dir_y = cv2.bitwise_and(part_dir_y, part_dir_y, mask=mask)

    # Construct structure tensor components
    Ixx = np.square(part_dir_x)
    Ixy = part_dir_x * part_dir_y
    Iyy = np.square(part_dir_x)

    # Run until area is fully painted
    while True:

        # Find the fill front of the fill area
        contours, hierarchy = cv2.findContours(image=patchthresh,
                                               mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_NONE)
        if len(contours) == 1:
            break

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

                # Construct the structure tensor for the point
                Ixx_patch = Ixx[y_start:y_end, x_start:x_end]
                Ixy_patch = Ixy[y_start:y_end, x_start:x_end]
                Iyy_patch = Iyy[y_start:y_end, x_start:x_end]

                Sxx = Ixx_patch[mask[y_start: y_end, x_start: x_end] == 255]
                Sxy = Ixy_patch[mask[y_start: y_end, x_start: x_end] == 255]
                Syy = Iyy_patch[mask[y_start: y_end, x_start: x_end] == 255]

                Sxx = np.mean(Sxx)
                Sxy = np.mean(Sxy)
                Syy = np.mean(Syy)

                S = np.array([[Sxx, Sxy], [Sxy, Syy]])

                # Calculate the structure tensor's eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(S)

                # Find the strongest eigenvector (orthogonal isothope)
                if eigenvalues[0] > eigenvalues[1]:
                    max_eigval_ind = 0
                else:
                    max_eigval_ind = 1
                max_eigvec = eigenvectors[:, max_eigval_ind]
                max_eigvec = np.array([-max_eigvec[1], max_eigvec[0]])
                max_eigvec *= eigenvalues[max_eigval_ind]

                # Calculate the data value
                d = abs(np.dot(max_eigvec, normal)) / 255

                # Calculate the point's priority
                priority = c * d

                # Find the point with highest priority
                if max_priority is None:
                    max_priority = priority
                    chosen_point = point
                    c_val = c

                if priority > max_priority:
                    max_priority = priority
                    chosen_point = point
                    c_val = c
        # Kernel indices for the chosen point
        k_inds = (chosen_point[1] - halfk, chosen_point[1] + halfk + 1,
                  chosen_point[0] - halfk, chosen_point[0] + halfk + 1)

        # Get the corresponding patch for the chosen point in the image
        fill_patch = patch[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]]
        fill_patchmask = patchthresh[k_inds[0]: k_inds[1],
                                     k_inds[2]: k_inds[3]] / 255
        fill_patchmask = fill_patchmask.astype('uint8')

        # Find the most similar patch in the rest of the image
        min_ssd = None
        sample_patch = None

        kernel = np.ones((ksize+1, ksize+1), np.uint8)
        overlap_check = cv2.dilate(255 - thresh, kernel, iterations=1)

        # Iterate through every possible patch in the source region
        for i in range(halfk, h//5 - halfk, 2):
            for j in range(w//2 + halfk, w - halfk, 2):

                # Make sure the patch doesn't overlap the empty area
                if overlap_check[i, j] == 255:
                    continue

                check_patch = img_lab[i - halfk: i + halfk + 1,
                                      j - halfk: j + halfk + 1]

                # Calculate the SSD (similarity) of the patches
                diff = cv2.bitwise_and(check_patch,
                                       check_patch,
                                       mask=fill_patchmask)
                diff = fill_patch - diff
                diff = np.square(diff)
                ssd = np.sum(diff)

                # Find the patch that minimises the SSD (distance)
                if min_ssd is None:
                    min_ssd = ssd
                    sample_patch = check_patch
                    continue

                if ssd < min_ssd:
                    min_ssd = ssd
                    sample_patch = check_patch

        # Extract the replacement content from the chosen example patch
        fill_patchmask = 1 - fill_patchmask
        sample_patch = cv2.bitwise_and(sample_patch,
                                       sample_patch,
                                       mask=fill_patchmask)

        # Replace the empty pixels with the values from the sample
        patch[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] += sample_patch
        sample_patch_grey = cv2.cvtColor(sample_patch, cv2.COLOR_BGR2GRAY)

        # Update the greyscale image
        patchgrey[k_inds[0]: k_inds[1],
                  k_inds[2]: k_inds[3]] += sample_patch_grey

        # Update the confidence values and mask
        conf_upd = fill_patchmask.astype('float64') * c_val
        patchconf[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] += conf_upd

        # Update thresholded images
        patchthresh[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] = 255
        thresh[y - halfk - 1: y + bh + halfk + 1,
               x - halfk - 1: x + bw + halfk + 1] = patchthresh

        # Update the partial derivatives
        patch_part_dir_x = cv2.Sobel(patchgrey[k_inds[0] - 1: k_inds[1] + 1,
                                               k_inds[2] - 1: k_inds[3] + 1],
                                     cv2.CV_64F, 1, 0, ksize=3)
        patch_part_dir_y = cv2.Sobel(patchgrey[k_inds[0] - 1: k_inds[1] + 1,
                                               k_inds[2] - 1: k_inds[3] + 1],
                                     cv2.CV_64F, 0, 1, ksize=3)
        patch_part_dir_x = patch_part_dir_x[1:-1, 1:-1]
        patch_part_dir_y = patch_part_dir_y[1:-1, 1:-1]

        # Update structure tensor components
        patch_Ixx = patch_part_dir_x ** 2
        patch_Ixy = patch_part_dir_x * patch_part_dir_y
        patch_Iyy = patch_part_dir_y ** 2

        Ixx[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] = patch_Ixx
        Ixy[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] = patch_Ixy
        Iyy[k_inds[0]: k_inds[1], k_inds[2]: k_inds[3]] = patch_Iyy

        # Use a dilated mask to prevent missing region interference
        mask = 255 - patchthresh
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = 255 - mask
        Ixx = cv2.bitwise_and(Ixx, Ixx, mask=mask)
        Ixy = cv2.bitwise_and(Ixy, Ixy, mask=mask)
        Iyy = cv2.bitwise_and(Iyy, Iyy, mask=mask)

    # Apply the fill and convert the image back to BGR
    img_lab[y - halfk - 1: y + bh + halfk + 1,
            x - halfk - 1: x + bw + halfk + 1] = patch
    res = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    return res


def kuwahara_filter(img, ksize, sigma):
    w, h = img.shape[:2]
    halfk = ksize//2

    # Padded the image for convolution
    padded = cv2.copyMakeBorder(img,
                                halfk, halfk, halfk, halfk,
                                cv2.BORDER_REFLECT)

    # Convert the image to LAB colour space
    img_lab = cv2.cvtColor(padded, cv2.COLOR_BGR2Lab).astype(np.float64)

    filtered_image = np.zeros(img.shape, dtype="float64")

    squared_l = img_lab[:, :, 0] ** 2

    # Create the Gaussian kernel
    kernel = cv2.getGaussianKernel(ksize, sigma)
    zs = np.zeros((halfk, 1))

    # Separate the kernel into left and right components
    kleft = kernel[:halfk+1]
    kright = kernel[halfk:]

    # Normalise each kernel component
    ker_norm = np.sum(kleft)
    kleft = kleft / ker_norm
    kright = kright / ker_norm

    # Pad the kernel for the separate kernel filter
    kleft = np.concatenate((kleft, zs))
    kright = np.concatenate((zs, kright))

    # Convolve the image for each quadrant of the filter to be considered
    segments = np.array([cv2.sepFilter2D(img_lab, -1, kleft, kleft),
                         cv2.sepFilter2D(img_lab, -1, kleft, kright),
                         cv2.sepFilter2D(img_lab, -1, kright, kleft),
                         cv2.sepFilter2D(img_lab, -1, kright, kright)])

    # Find the mean luminance for each quadrant for each pixel
    mean_l = np.array([segments[0][:, :, 0],
                       segments[1][:, :, 0],
                       segments[2][:, :, 0],
                       segments[3][:, :, 0]])

    # Find the standard deviation for each quadrant for each pixel
    std_l = [cv2.sepFilter2D(squared_l, -1, kleft, kleft),
             cv2.sepFilter2D(squared_l, -1, kleft, kright),
             cv2.sepFilter2D(squared_l, -1, kright, kleft),
             cv2.sepFilter2D(squared_l, -1, kright, kright)]

    std_l = [std_l[0] - (mean_l[0] ** 2),
             std_l[1] - (mean_l[1] ** 2),
             std_l[2] - (mean_l[2] ** 2),
             std_l[3] - (mean_l[3] ** 2)]

    std_l = np.array([np.sqrt(np.abs(std_l[0])),
                      np.sqrt(np.abs(std_l[1])),
                      np.sqrt(np.abs(std_l[2])),
                      np.sqrt(np.abs(std_l[3]))])

    # Calculate the quadrant weights
    weights = np.array([1 / (1 + std_l[0]),
                        1 / (1 + std_l[1]),
                        1 / (1 + std_l[2]),
                        1 / (1 + std_l[3])])

    # Convolve the filter
    for y in range(halfk, h + halfk):
        for x in range(halfk, w + halfk):
            wsum = 0
            # Compute the resulting pixel colour
            for i in range(4):
                wsum += weights[i][y, x]
                quad_val = segments[i][y, x] * weights[i][y, x]
                filtered_image[y-halfk][x-halfk] += quad_val
            filtered_image[y-halfk][x-halfk] /= wsum

    # Convert the image back to BGR colour space
    filtered_image = filtered_image.astype(np.uint8)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_Lab2BGR)
    return filtered_image


def process(img):
    w, h = img.shape[:2]

    # Manually defined image corners
    corners = np.float32([[9, 15], [233, 5], [30, 235], [249, 227]])
    bounds = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Warp the perspective so the corners fit the image
    M = cv2.getPerspectiveTransform(corners, bounds)
    warp = cv2.warpPerspective(img, M, (w, h))

    # Kuwahara filter for salt & pepper noise
    kuwa = kuwahara_filter(warp, 5, 1.5)

    # Non-local means for Gaussian noise
    denoise = cv2.fastNlMeansDenoisingColored(kuwa, None, 10, 10, 11, 27)

    # Additional bilateral filter smoothing
    bilat = cv2.bilateralFilter(denoise, 9, 20, 20)

    # Histogram equalisation for contrast enhancement
    img_YCrCb = cv2.cvtColor(bilat, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(10, 10))
    img_YCrCb[:, :, 0] = clahe.apply(img_YCrCb[:, :, 0])
    contrast = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2BGR)

    # Kuwahara filter for residual noise removal
    kuwa2 = kuwahara_filter(contrast, 5, 0.25)

    # Inpainting the missing region
    paint = exemplar_paint(kuwa2, 9)

    # Create a mask of the missing region
    gray = cv2.cvtColor(bilat, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(bilat.shape[:2], dtype="uint8")
    mask[gray < 20] = 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Add the inpainting to the image with blur for smoothing
    paint2 = paint.copy()
    paint2 = cv2.medianBlur(paint2, 5)
    paint2 = cv2.GaussianBlur(paint2, (3, 3), 2)
    paint[mask == 255] = paint2[mask == 255]

    # Sharpen using the Laplacian
    laplace = cv2.Laplacian(paint, cv2.CV_64F, ksize=3)
    imgf64 = np.float64(paint)
    sharp = cv2.subtract(imgf64, laplace * 1.1)
    sharp = np.clip(sharp, 0, 255).astype('uint8')

    # Final bilateral filter denoising
    denoise2 = cv2.bilateralFilter(sharp, 9, 50, 50)

    return denoise2


if __name__ == "__main__":
    # Find the images to be processed and results directory
    path = sys.argv[1]
    if not os.path.exists("./Results"):
        os.makedirs("./Results")

    # Process every file in the target directory
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.jpg' not in file:
                continue

            # Find the target file
            file_path = os.path.join(root, file)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            # Process the image
            res = process(img)

            # Save the processed image to the results directory
            cv2.imwrite(os.path.join("./Results", file), res)
