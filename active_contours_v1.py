import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

def getBBoxContour(bbox, N=10):
    (x1, y1), (x2, y2) = bbox
    x = np.array([np.linspace(x1, x1, N // 4), np.linspace(x1, x2, N // 4),
         np.linspace(x2, x2, N // 4), np.linspace(x2, x1, N // 4)]).flatten()
    y = np.array([np.linspace(y1, y2, N // 4), np.linspace(y2, y2, N // 4),
         np.linspace(y2, y1, N // 4), np.linspace(y1, y1, N // 4)]).flatten()
    return np.array([x, y])

def getCircleContour(centre=(0, 0), radius=(1, 1), N=200):
    """
    以參數方程的形式，獲取n個離散點圍成的圓形/橢圓形輪廓
    輸入：中心centre=（x0, y0）, 半軸長radius=(a, b)， 離散點數N
    輸出：由離散點座標(x, y)組成的2xN矩陣
    """
    t = np.linspace(0, 2 * np.pi, N)
    x = centre[0] + radius[0] * np.cos(t)
    y = centre[1] + radius[1] * np.sin(t)
    return np.array([x, y])

def plt_setting(img, init, snake):
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.plot(init[0, :], init[1,:], '--r', lw=3)
    print('init[:, 0]', init[:, 0])
    print('init[:, 1]', init[:, 1])
    plt.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    plt.xticks([]), plt.yticks([]), plt.axis("off")
    plt.show()

def act_contours(img_filename, bbox, N):
    src = io.imread(img_filename)
    img = rgb2gray(src)

    init = getBBoxContour(bbox, N)
    print(init)
    snake = active_contour(gaussian(img, 3), snake=init, alpha=0.1, beta=1, gamma=0.01, w_line=0, w_edge=10)
    # snake = active_contour(gaussian(img, 3), snake=init, alpha=0.1, beta=1, gamma=0.01, w_line=0, w_edge=10)
    print(snake)
    print(init)
    print(img.shape)
    plt_setting(img, init, snake)

if __name__ == '__main__':
    '''
    img = data.astronaut()
    # img = io.imread("腎臟.png") # 讀入圖像
    img = rgb2gray(img) # 灰度化

    # init = getCircleContour((455, 340), (370, 250), N=80)
    # init = getCircleContour((220, 100), (100, 100), N=400)
    t = np.linspace(0, 2 * np.pi, 400)  # 參數t, [0,2π]
    r = 100 + 100 * np.sin(t)
    c = 220 + 100 * np.cos(t)
    init = np.array([r, c]).T
    # init = init.T
    print('init', init.T.shape)

    # Snake模型迭代輸出
    # snake = active_contour(gaussian(img,3), snake=init, alpha=0.1, beta=1, gamma=0.01, w_line=0, w_edge=10)
    snake = active_contour(gaussian(img, 3, preserve_range=False), snake=init, alpha=0.015, beta=10, gamma=0.001, w_line=0, w_edge=10)

    print('snake', snake.shape)
    # print('snake', snake)
    # 繪圖顯示
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.plot(init[:, 0], init[:, 1], '--r', lw=3)
    plt.plot(snake[0, :], snake[1, :], '-b', lw=3)
    plt.xticks([]), plt.yticks([]), plt.axis("off")
    plt.show()
    '''
    img = data.astronaut()
    # img = rgb2gray(img)

    s = np.linspace(0, 2 * np.pi, 100)
    r = 100 + 100 * np.sin(s)
    c = 220 + 100 * np.cos(s)
    init = np.array([r, c]).T
    print('init', init.shape)
    # snake = active_contour(gaussian(img, 3, preserve_range=False),
    #                        init, alpha=0.015, beta=10, gamma=0.001, w_line=10, w_edge=5)
    # snake = active_contour(gaussian(img, 3),
    #                        init, alpha=0.015, beta=10, gamma=0.001, w_line=0, w_edge=5)
    snake = active_contour(gaussian(img, 3), init, alpha=0.1, beta=1, gamma=0.01, w_line=0, w_edge=10)
    # snake = active_contour(img, init, alpha=0.015, beta=10, gamma=0.001)
    print('snake', snake.shape)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()