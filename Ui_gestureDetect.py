# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Hasee\Desktop\计算机视觉\lastwork\gesture-recognition-master\gestureDetect.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1081, 802)
        self.videoLabel = QtWidgets.QLabel(Form)
        self.videoLabel.setGeometry(QtCore.QRect(40, 30, 1011, 621))
        self.videoLabel.setObjectName("videoLabel")
        self.fingercount = QtWidgets.QPushButton(Form)
        self.fingercount.setGeometry(QtCore.QRect(500, 680, 93, 28))
        self.fingercount.setObjectName("fingercount")
        self.stopButton = QtWidgets.QPushButton(Form)
        self.stopButton.setGeometry(QtCore.QRect(680, 680, 93, 28))
        self.stopButton.setObjectName("stopButton")
        self.virtualMouse = QtWidgets.QPushButton(Form)
        self.virtualMouse.setGeometry(QtCore.QRect(300, 680, 93, 28))
        self.virtualMouse.setObjectName("virtualMouse")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.videoLabel.setText(_translate("Form", "view"))
        self.fingercount.setText(_translate("Form", "fingercount"))
        self.stopButton.setText(_translate("Form", "stop"))
        self.virtualMouse.setText(_translate("Form", "mouse"))
