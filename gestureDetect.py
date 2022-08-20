import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QTimer
import numpy as np
import cv2
from Ui_gestureDetect import *
from recognize import *
from mouse import *

num_frames = 0

rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))


class Video():
    def __init__(self, capture):
        self.capture = capture
        self.currentFrame = np.array([])

    def captureFrame(self):
        ret, readFrame = self.capture.read()
        return readFrame

    def captureNextFrame(self):
        global num_frames
        ret, readFrame = self.capture.read()
        (readFrame, thresholded) = recognizecore(readFrame, num_frames)
        if thresholded is not None:
            cv2.imshow("Thesholded", thresholded)
        num_frames += 1
        if (ret == True):
            self.currentFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2RGB)

    def captureNextFrameMouse(self):
        global rectList
        ret, readFrame = self.capture.read()
        readFrame = gestureMouse(readFrame, rectList)
        # cv2.imshow("readFrame",readFrame)
        if (ret == True):
            self.currentFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2RGB)

    def convertFrame(self):
        try:
            height, width = self.currentFrame.shape[:2]
            img = QImage(self.currentFrame, width,
                         height, QImage.Format_RGB888)
            img = QPixmap.fromImage(img)
            self.previousFrame = self.currentFrame
            return img
        except:
            return None


class MyMainForm(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        self.video = Video(cap)
        self._timer = QTimer(self)
        self.ret, self.capturedFrame = self.video.capture.read()
        self.fingercount.clicked.connect(self.openfingercount)
        self.stopButton.clicked.connect(self.closecamera)
        self.virtualMouse.clicked.connect(self.openMouse)

    def openfingercount(self):
        self._timer.timeout.connect(self.play)
        self._timer.start(27)
        self.update()
        self.videoFrame = self.videoLabel
        # self.videoFrame.setAlignment(Qt.AlignCenter) # 消除布局中的空隙，让两个控件紧紧挨在一起。
        # self.setCentralWidget(self.videoFrame)  # 会充满整个界面

    def play(self):
        try:
            self.video.captureNextFrame()
            self.videoFrame.setPixmap(self.video.convertFrame())
            self.videoFrame.setScaledContents(True)  # 自适应窗口大小
        except TypeError:
            print('No Frame')

    def openMouse(self):
        self._timer.timeout.connect(self.mouseplay)
        self._timer.start(27)
        self.update()
        self.videoFrame = self.videoLabel
        # self.videoFrame.setAlignment(Qt.AlignCenter) # 消除布局中的空隙，让两个控件紧紧挨在一起。
        # self.setCentralWidget(self.videoFrame)  # 会充满整个界面

    def mouseplay(self):
        try:
            self.video.captureNextFrameMouse()
            self.videoFrame.setPixmap(self.video.convertFrame())
            self.videoFrame.setScaledContents(True)  # 自适应窗口大小
        except TypeError:
            print('No Frame')

    def closecamera(self):

        cv2.VideoCapture(0).release()
        self._timer.stop()


if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
