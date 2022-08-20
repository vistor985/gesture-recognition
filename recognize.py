# ------------------------------------------------------------
# SEGMENT, RECOGNIZE and COUNT fingers from a video sequence
# ------------------------------------------------------------

# organize imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

# global variables
bg = None

# --------------------------------------------------
# To find the running average over the background
# --------------------------------------------------


def run_avg(image, accumWeight):
    global bg
    # 初始化背景
    if bg is None:
        bg = image.copy().astype("float")
        return

    # 计算加权平均值，累积并更新背景
    cv2.accumulateWeighted(image, bg, accumWeight)

# ---------------------------------------------
# To segment the region of hand in the image
# ---------------------------------------------


def segment(image, threshold=25):
    global bg
    # 找到背景和当前帧之间的绝对差异
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # 阈值差异图像，以便我们获得前景
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # 获取阈值图像中的轮廓
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有检测到轮廓直接返回
    if len(cnts) == 0:
        return
    else:
        # 根据轮廓面积，得到最大轮廓即手
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

# --------------------------------------------------------------
# To count the number of fingers in the segmented hand region
# --------------------------------------------------------------


def count(thresholded, segmented):
    # 找到分段手区域的凸包
    chull = cv2.convexHull(segmented)

    # 在凸包中找到最极端的点
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    # 找到手掌的中心
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # 找到手掌中心之间的最大欧几里得距离和凸包的最极端点
    distance = pairwise.euclidean_distances(
        [(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # 用获得的最大欧几里德距离的 80% 计算圆的半径
    radius = int(0.8 * maximum_distance)

    # 找到圆的周长
    circumference = (2 * np.pi * radius)

    # 取出有兴趣的圆形区域--手掌和手指
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    # 绘制圆形 ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # 使用圆形 ROI 作为掩码在阈值手之间进行按位与运算，该掩码给出使用阈值手图像上的掩码获得的切割
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # 计算圆形 ROI 中的轮廓
    (cnts, _) = cv2.findContours(circular_roi.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 初始化手指计数
    count = 0

    # 遍历找到的轮廓
    for c in cnts:
        # 计算轮廓的边界框
        (x, y, w, h) = cv2.boundingRect(c)

        # 仅在以下情况增加手指数
        # 1.轮廓区域不是手腕（底部区域）
        # 2. 沿轮廓的点数不超过圆形 ROI 周长的 25%
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count


def recognizecore(frame, num_frames):
    # 初始化累积权重
    accumWeight = 0.5

    # 感兴趣的区域 (ROI) 坐标
    top, right, bottom, left = 10, 350, 225, 590

    
    # 调整框架大小
    frame = imutils.resize(frame, width=700)

    # 翻转得到的图像，使其不是镜像图
    frame = cv2.flip(frame, 1)

    # 复制图像
    clone = frame.copy()

    # 用于记录thresholded后面输出给外部
    showthre = None

    # 获取框架的高度和宽度
    (height, width) = frame.shape[:2]

    # 得到 ROI（感兴趣区域）
    roi = frame[top:bottom, right:left]

    # 将ROI转为灰度并模糊它
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 前30帧用于构建背景
    # 以便我们的加权平均模型得到校准
    if num_frames < 30:
        run_avg(gray, accumWeight)
        if num_frames == 1:
            print("[STATUS] please wait! calibrating...")
        elif num_frames == 29:
            print("[STATUS] calibration successfull...")
    else:
        # 分割手部区域
        hand = segment(gray)

        # 检查手部区域是否被分割出来
        if hand is not None:
            # 得到返回的两个参数
            (thresholded, segmented) = hand

            # 绘制分割区域
            cv2.drawContours(
                clone, [segmented + (right, top)], -1, (0, 0, 255))

            # 计算手指数量
            fingers = count(thresholded, segmented)

            cv2.putText(clone, str(fingers), (70, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 记录下二值图像用于return
            showthre = thresholded.copy()

    # 画出分割的手
    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

    return clone, showthre
