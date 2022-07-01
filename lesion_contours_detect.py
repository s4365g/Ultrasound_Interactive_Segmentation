import os
import glob
import math
import numpy as np
import time
import cv2
from skimage.morphology import skeletonize, thin
from skimage.segmentation import active_contour
from PIL import Image
from matplotlib import pyplot as plt
def build_RD_flt(kernel_size):
    if (kernel_size % 2) == 1:
        mid = kernel_size // 2
    RD1_flt = np.zeros((kernel_size, kernel_size))
    RD2_flt = np.zeros((kernel_size, kernel_size))
    RD3_flt = np.zeros((kernel_size, kernel_size))
    RD4_flt = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i < mid:
                RD1_flt[i, j] = -1
            elif i > mid:
                RD1_flt[i, j] = 1
            if j + i < kernel_size - 1:
                RD2_flt[i, j] = -1
            elif j + i > kernel_size - 1:
                RD2_flt[i, j] = 1
            if j < mid:
                RD3_flt[i, j] = -1
            elif j > mid:
                RD3_flt[i, j] = 1
            if j - i > 0:
                RD4_flt[i, j] = 1
            elif j - i < 0:
                RD4_flt[i, j] = -1
    return RD1_flt, RD2_flt, RD3_flt, RD4_flt

def atm_flt(array_2d, flt, alpha=0.2):
    ## k=kernel_size
    h, w = array_2d.shape
    k, _ = flt.shape
    hb, he = k//2, h - k//2
    wb, we = k//2, w - k//2
    out_img = np.zeros((h,w))
    for i in range(hb, he):
        for j in range(wb, we):
            Is = np.multiply(array_2d[i-k//2 : i+k//2+1, j-k//2 : j+k//2+1], flt)

            mtz = Is[flt > 0].astype(np.float32).copy() # more than zero
            ltz = Is[flt < 0].astype(np.float32).copy() # less than zero
            mtz_sorted = np.sort(mtz.flatten())
            ltz_sorted = np.sort(ltz.flatten())
            size_len = k*math.ceil(k/2-1)
            AM1 = np.mean(mtz_sorted[int(size_len*alpha):int(size_len+1-size_len*alpha+1)])
            AM2 = np.mean(ltz_sorted[int(size_len*alpha):int(size_len+1-size_len*alpha+1)])
            abs_value = abs(abs(AM1)-abs(AM2))
            out_img[i, j] = abs_value
    return out_img

def Region_diff_img_v1(blur_img, k, save_img=False):
    RD1_flt, RD2_flt, RD3_flt, RD4_flt = build_RD_flt(k)

    RD1_atm_img = atm_flt(blur_img, RD1_flt, alpha=0)
    RD2_atm_img = atm_flt(blur_img, RD2_flt, alpha=0)
    RD3_atm_img = atm_flt(blur_img, RD3_flt, alpha=0)
    RD4_atm_img = atm_flt(blur_img, RD4_flt, alpha=0)

    RD12_max = np.maximum(RD1_atm_img, RD2_atm_img)
    RD34_max = np.maximum(RD3_atm_img, RD4_atm_img)
    RD_img = np.maximum(RD12_max, RD34_max)
    RD_img = cv2.convertScaleAbs(RD_img)
    if save_img:
        cv2.imwrite(save_img, RD_img)
    return RD_img

def Region_diff_img_v2(blur_img, k, save_img=False):
    RD1_flt, RD2_flt, RD3_flt, RD4_flt = build_RD_flt(k)
    RD1_atm_img = cv2.filter2D(blur_img.astype(np.float32), ddepth=-1, kernel=RD1_flt)
    RD2_atm_img = cv2.filter2D(blur_img.astype(np.float32), ddepth=-1, kernel=RD2_flt)
    RD3_atm_img = cv2.filter2D(blur_img.astype(np.float32), ddepth=-1, kernel=RD3_flt)
    RD4_atm_img = cv2.filter2D(blur_img.astype(np.float32), ddepth=-1, kernel=RD4_flt)
    RD12_max = np.maximum(abs(RD1_atm_img), abs(RD2_atm_img))
    RD34_max = np.maximum(abs(RD3_atm_img), abs(RD4_atm_img))
    RD_img = np.maximum(RD12_max, RD34_max)
    RD_img = cv2.convertScaleAbs(RD_img)
    if save_img:
        cv2.imwrite(save_img, RD_img)
    return RD_img

def Region_diff_img_v3(blur_img, save_img=False):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    img_sobel_x = cv2.filter2D(blur_img.astype(np.float64), ddepth=-1, kernel=sobel_x)
    img_sobel_x = abs(img_sobel_x)

    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_sobel_y = cv2.filter2D(blur_img.astype(np.float64), ddepth=-1, kernel=sobel_y)
    img_sobel_y = abs(img_sobel_y)

    sobel_d1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])  # diagonel style maxtrix top-left to bottom-right is 0
    img_sobel_d1 = cv2.filter2D(blur_img.astype(np.float64), ddepth=-1, kernel=sobel_d1)
    img_sobel_d1 = abs(img_sobel_d1)

    sobel_d2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])  # diagonel style maxtrix bottom-left to top-right is 0
    img_sobel_d2 = cv2.filter2D(blur_img.astype(np.float64), ddepth=-1, kernel=sobel_d2)
    img_sobel_d2 = abs(img_sobel_d2)

    RD_1_img = np.maximum(img_sobel_x, img_sobel_y)
    RD_2_img = np.maximum(img_sobel_d1, img_sobel_d2)
    RD_img = np.maximum(RD_1_img, RD_2_img)
    RD_img = cv2.convertScaleAbs(RD_img)
    if save_img:
        cv2.imwrite(save_img, RD_img)
    return RD_img

def lesion_contours(img, blur_k=7, bin_thres=20):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.medianBlur(gray_img, blur_k)
    RD_img = Region_diff_img_v2(blur_img, 3)
    ret, binary_img = cv2.threshold(RD_img, bin_thres, 255, cv2.THRESH_BINARY)
    k = 2
    kernel = np.ones((k, k), np.float64)
    binary_img = cv2.morphologyEx(binary_img.astype(np.float64), cv2.MORPH_OPEN, kernel)
    out_img = cv2.erode(binary_img.astype(np.float64), kernel, iterations=1)
    contours, hierarchy = cv2.findContours(out_img.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_dropped = tuple()
    hierarchy_dropped = list()
    mapping_id = dict()
    count = 0
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) >= 5:
            contours_dropped += (cnt,)
            hierarchy_dropped.append(list(hierarchy[0, i, :]))
            mapping_id[count] = i
            count += 1
    hierarchy_dropped = np.array([hierarchy_dropped])
    return contours_dropped, hierarchy_dropped, mapping_id

def lesion_contours_andy(img, shift_axis, blur_k=7, bin_thres=20):
    titles = []
    images = []
    kernal_gauss = (3, 3)
    kernel_open = np.ones((3,3),np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_img.shape)
    if min(gray_img.shape) > max(kernal_gauss) * 40:
        gray_img = cv2.GaussianBlur(gray_img, kernal_gauss, 0)

    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).ravel()
    titles.append(['Histogram'])
    images.append([hist])

    # version 1 Histogram  Equalization
    # v1_gray_img = gray_img.copy()
    # v1_eq = cv2.equalizeHist(v1_gray_img)
    # _, v1_thresh_img = cv2.threshold(v1_eq, 127, 255, cv2.THRESH_BINARY)
    # v1_opening = cv2.morphologyEx(v1_thresh_img, cv2.MORPH_OPEN, kernel_open)
    # v1_closing = cv2.morphologyEx(v1_opening, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    #
    # titles.append(['V1 Gray Image', 'V1 EqualizeHist', 'V1 BINARY', 'V1 OPENING', 'V1 CLOSING'])
    # images.append([v1_gray_img, v1_eq, v1_thresh_img, v1_opening, v1_closing])


    # version 2 Otsu's Method
    v2_gray_img = gray_img.copy()
    _, v2_THRESH_OTSU_img = cv2.threshold(v2_gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # find main item intensity
    rect_size_x = int(gray_img.shape[1] / 3)
    rect_size_y = int(gray_img.shape[0] / 3)
    rect_center = (np.asarray(v2_gray_img.shape)/2)
    rect_point1 = (int(rect_center[1] - rect_size_x/2), int(rect_center[0] - rect_size_y/2))
    rect_point2 = (int(rect_center[1] + rect_size_x / 2), int(rect_center[0] + rect_size_y / 2))
    v2_THRESH_OTSU_bgr_img = cv2.cvtColor(v2_THRESH_OTSU_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(v2_THRESH_OTSU_bgr_img, rect_point1, rect_point2, (255, 0, 0), 0)

    main_obj = v2_THRESH_OTSU_img[rect_point1[1]:rect_point2[1], rect_point1[0]:rect_point2[0]]
    num_0 = np.sum(main_obj == 0)
    num_255 = np.sum(main_obj == 255)
    if num_0 > num_255:
        v2_THRESH_OTSU_img = cv2.bitwise_not(v2_THRESH_OTSU_img)

    # morphology
    v2_opening = cv2.morphologyEx(v2_THRESH_OTSU_img, cv2.MORPH_OPEN, kernel_open, iterations= 3 if min(gray_img.shape) > 100 else 1)
    v2_closing = cv2.morphologyEx(v2_opening, cv2.MORPH_CLOSE, kernel_close, iterations= 2 if min(gray_img.shape) > 100 else 1)

    # Connected Component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(v2_closing, connectivity=4)
    output_connectedComponent = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    main_index = np.argmax(stats[1:,-1])
    for i in range(1, num_labels):
        if i == main_index + 1:
            mask = labels == i
            output_connectedComponent[:, :, 0][mask] = 255
            # output_connectedComponent[:, :, 1][mask] = 255
            # output_connectedComponent[:, :, 2][mask] = 255

    # find contour
    contours, hierarchy = cv2.findContours(output_connectedComponent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v2_gray_img_contours = cv2.cvtColor(v2_gray_img, cv2.COLOR_GRAY2BGR)
    print("contours: ", contours)
    print("type(contours): ",type(contours))
    cv2.drawContours(v2_gray_img_contours, contours, -1, (255,0,0), 1)

    titles.append(['V2 Gray Image', 'V2 OTSU BINARY', 'V2 OPENING', 'V2 CLOSING', 'V2 Connected Component', 'V2 contours'])
    images.append([v2_gray_img, v2_THRESH_OTSU_bgr_img, v2_opening, v2_closing, output_connectedComponent, v2_gray_img_contours])
    subplot_figure(titles, images)

    # Calculate output contour
    contours_rawImg = tuple((np.array(contours) + shift_axis))
    return contours_rawImg

def subplot_figure(titles, images):
    plt.figure(figsize=(12, 10))
    rows = len(max(titles, key=len))
    cols = len(titles)
    for x in range(len(titles)):        # x axis
        for y in range(len(titles[x])): # y axis
            if x == 0:
                plt.subplot(rows, cols, cols * y + x + 1), plt.bar(range(1, 257), images[x][y])
            else:
                plt.subplot(rows, cols, cols * y + x + 1), plt.imshow(images[x][y], 'gray')
            plt.title(titles[x][y], fontsize=8)
            plt.xticks([]), plt.yticks([])

    plt.show()
def lesion_binary_contours(img, pre_bin = 150, blur_k=7, bin_thres=20):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray_img = cv2.threshold(gray_img, pre_bin, 255, cv2.THRESH_BINARY)
    blur_img = cv2.medianBlur(gray_img, blur_k)
    RD_img = Region_diff_img_v2(blur_img, 3)
    ret, binary_img = cv2.threshold(RD_img, bin_thres, 255, cv2.THRESH_BINARY)
    k = 2
    kernel = np.ones((k, k), np.float64)
    binary_img = cv2.morphologyEx(binary_img.astype(np.float64), cv2.MORPH_OPEN, kernel)
    out_img = cv2.erode(binary_img.astype(np.float64), kernel, iterations=1)
    contours, hierarchy = cv2.findContours(out_img.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_dropped = tuple()
    hierarchy_dropped = list()
    mapping_id = dict()
    count = 0
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) >= 5:
            contours_dropped += (cnt,)
            hierarchy_dropped.append(list(hierarchy[0, i, :]))
            mapping_id[count] = i
            count += 1
    hierarchy_dropped = np.array([hierarchy_dropped])
    return contours_dropped, hierarchy_dropped, mapping_id

def check_contour(point, contours, hierarchy, mapping_id):
    for i, cnt in enumerate(contours):
        dst = cv2.pointPolygonTest(cnt, point, True)
        if dst >= 0 and hierarchy[0, i, 3] == -1:
            parent = mapping_id[i]
            print('this is parent', parent, hierarchy[0, i, :], cv2.contourArea(cnt), len(cnt))
            print('type(cnt)', type(cnt))
            break
    for i, cnt in enumerate(contours):
        if mapping_id[i] == parent:
            print('this is parent', i, hierarchy[0, i, :])
        if hierarchy[0, i, 3] == parent:
            print('this is children', i, hierarchy[0, i, :])

def liver_binary_small_test(filename):
    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret1, th1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_TOZERO)

    test_3 = gray_img.copy()
    plt.hist(test_3.ravel(), 256)
    plt.title('gray_histogram')
    plt.show()
    his_list = test_3.reshape(-1)
    for i in [0, 1]:
        his_list = np.delete(his_list, np.where(his_list==i))
    std = np.std(his_list)
    mean = np.mean(his_list)
    print(std, mean)

    test_4 = gray_img.copy()
    test_4[np.logical_and(test_4 >= 50, test_4 <= 100)] = 0
    plt.imshow(test_4, 'gray')
    plt.title('INV bandpass filter')
    plt.show()
    plt.hist(test_4.ravel(), 256)
    plt.title('test_4:INV bandpass filter')
    plt.show()

    test_5 = gray_img.copy()
    test_5[test_5 >= mean-std] = 255
    plt.imshow(test_5, 'gray')
    plt.title('lowpass filter')
    plt.show()
    plt.hist(test_5.ravel(), 256)
    plt.title('test_5:lowpass filter')
    plt.show()
    cv2.imwrite('lowpass_filter.png', test_5)

    test_6 = gray_img.copy()
    test_6[test_6 <= mean+std] = 0
    plt.imshow(test_6, 'gray')
    plt.title('highpass filter')
    plt.show()
    plt.hist(test_6.ravel(), 256)
    plt.title('test_6:highpass filter')
    plt.show()
    cv2.imwrite('highpass_filter.png', test_6)

if __name__ == '__main__':
    filename_list = glob.glob(r"./image_crop/*.jpg")
    for filename in filename_list:
        im = cv2.imread(filename)
        lesion_contours_andy(im, (0, 0))



    # contours_dropped, hierarchy_dropped, mapping_id = lesion_contours(im)
    # point = (242, 268)
    # for i, cnt in enumerate(contours_dropped):
    #     dst = cv2.pointPolygonTest(cnt, point, True)
    #     if dst >= 0 and hierarchy_dropped[0, i, 3] == -1:
    #         # parent = mapping_id[i]
    #         # print('this is parent', parent, hierarchy_dropped[0, i, :], cv2.contourArea(cnt), len(cnt))
    #         # print('type(cnt)', type(cnt))
    #         pick_cnt = cnt
    #         break
    # im_out = cv2.drawContours(im, [pick_cnt], -1, (255, 0, 0), 1)
    # cv2.imwrite('draw_contours_dropped.jpg', im_out)
    #
    # im_out = cv2.drawContours(im, contours_dropped, -1, (255, 0, 0), 1)
    # cv2.imwrite('draw_contours_dropped_all.jpg', im_out)

