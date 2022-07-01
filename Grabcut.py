import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import io
from skimage import data
# from skimage.color import rgb2gray

def plt_mask_setting(src, mask):
    mask2 = np.zeros(src.shape, np.uint8)
    mask2[:, :, 0] = np.where((mask == 0), 0, 255).astype('uint8')
    mask2[:, :, 1] = np.where((mask == 0), 0, 255).astype('uint8')
    mask2[:, :, 2] = np.where((mask == 0), 0, 255).astype('uint8')

    plt.figure()
    plt.imshow(src)
    plt.show()

    plt.figure()
    plt.imshow(mask2)
    plt.show()

    img = src * mask[:, :, np.newaxis]
    plt.figure()
    plt.imshow(img)
    plt.show()

def plt_contours_setting(img):

    plt.figure()
    plt.imshow(img)
    plt.show()


def Grabcut(filename, bbox):
    src = cv2.imread(filename)
    JET_src = cv2.applyColorMap(src, cv2.COLORMAP_JET)
    plt_contours_setting(JET_src)
    # JET_src = cv2.applyColorMap(src, cv2.COLORMAP_HOT)
    print('filename', filename, src.shape)
    mask = np.zeros(src.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    (x1, y1), (x2, y2) = bbox
    rect = (x1, y1, x2-x1, y2-y1) # x1, y1, w, h
    # rect_to_mask = mask.copy()

    # plt_setting(src, rect_to_mask)
    print('rect', rect)
    cv2.grabCut(JET_src, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # plt_setting(src, mask2)
    kernel = np.ones((8, 8), np.uint8)
    dilate_mask = cv2.dilate(mask2, kernel, iterations=1)
    kernel = np.ones((9, 9), np.uint8)
    mask2 = cv2.erode(dilate_mask, kernel, iterations=1)
    # plt_mask_setting(src, mask2)
    cnts, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    out_img = src.copy()
    cv2.drawContours(out_img, cnts, -1, (0, 255, 0), 5)
    plt_contours_setting(out_img)

    #
    crop_src = src[x1:x2, y1:y2, :]
    max = np.amax(crop_src)
    min = np.amin(crop_src)
    enhance_src = np.where(src>=max, 255, src).astype('uint8')
    enhance_src = np.where(enhance_src < min, 0, src).astype('uint8')
    enhance_src = np.where((enhance_src >= min) & (enhance_src < max), (255/(max-min))*(enhance_src-min), src).astype('uint8')
    JET_masked_src = cv2.applyColorMap(enhance_src, cv2.COLORMAP_JET)
    plt_contours_setting(JET_masked_src)
    mask_test = np.zeros(src.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(JET_masked_src, mask_test, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask3 = np.where((mask_test == 2) | (mask_test == 0), 0, 1).astype('uint8')
    kernel = np.ones((8, 8), np.uint8)
    dilate_mask = cv2.dilate(mask3, kernel, iterations=1)
    kernel = np.ones((9, 9), np.uint8)
    mask3 = cv2.erode(dilate_mask, kernel, iterations=1)
    cnts, hierarchy = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    out_img = cv2.drawContours(src, cnts, -1, (0, 255, 0), 5)
    plt_contours_setting(out_img)

def Grabcut_v2(filename, bbox):
    src = cv2.imread(filename)
    # JET_src = cv2.applyColorMap(src, cv2.COLORMAP_JET)
    JET_src = cv2.applyColorMap(src, cv2.COLORMAP_HOT)
    print('filename', filename, src.shape)
    # src = io.imread(filename)
    mask = np.zeros(src.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    (x1, y1), (x2, y2) = bbox
    # (y1, x1), (y2, x2) = bbox
    # rect = (min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2))
    rect = (x1, y1, x2-x1, y2-y1)
    # rect_to_mask = mask.copy()
    # rect_to_mask = cv2.rectangle(rect_to_mask, (x1, y1), (x2, y2), 1, -1)
    # plt_setting(src, rect_to_mask)
    print('rect', rect)
    cv2.grabCut(JET_src, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # plt_setting(src, mask2)
    kernel = np.ones((8, 8), np.uint8)
    dilate_mask = cv2.dilate(mask2, kernel, iterations=1)
    kernel = np.ones((9, 9), np.uint8)
    mask2 = cv2.erode(dilate_mask, kernel, iterations=1)
    # plt_mask_setting(src, mask2)
    cnts, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    out_img = cv2.drawContours(src, cnts, -1, (0, 255, 0), 5)
    plt_contours_setting(out_img)
    # edged_img = cv2.drawContours(src, cnts, -1 (0, 255, 0), 2)
    # plt_setting(src, edged_img)




if __name__ == '__main__':
    src = data.astronaut()
    # src = rgb2gray(src)  # 灰度化

    # img = cv2.GaussianBlur(src, (3, 3), 5)

    mask = np.zeros(src.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # rect = (10, 150, 400, 540)
    rect = (150, 20, 200, 200)
    # rect = (350, 0, 480, 320)
    cv2.grabCut(src, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    kernel = np.ones((8,8), np.uint8)
    dilate_mask = cv2.dilate(mask2, kernel, iterations = 1)
    kernel = np.ones((9, 9), np.uint8)
    mask2 = cv2.erode(dilate_mask, kernel, iterations=1)
    img = src * mask2[:, :, np.newaxis]

    plt.imshow(src)
    plt.figure()
    plt.imshow(img), plt.colorbar(), plt.show()