# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SelectCorrespondences.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_SelectCorrespondences(object):
    def setupUi(self, SelectCorrespondences):
        if not SelectCorrespondences.objectName():
            SelectCorrespondences.setObjectName(u"SelectCorrespondences")
        SelectCorrespondences.setWindowModality(Qt.NonModal)
        SelectCorrespondences.resize(1087, 706)
        SelectCorrespondences.setWindowTitle(u"SelectCorrespondences")
        self.vl_base = QVBoxLayout(SelectCorrespondences)
        self.vl_base.setObjectName(u"vl_base")
        self.splitter_base = QSplitter(SelectCorrespondences)
        self.splitter_base.setObjectName(u"splitter_base")
        self.splitter_base.setOrientation(Qt.Horizontal)
        self.layoutWidget_base = QWidget(self.splitter_base)
        self.layoutWidget_base.setObjectName(u"layoutWidget_base")
        self.vl_operations = QVBoxLayout(self.layoutWidget_base)
        self.vl_operations.setObjectName(u"vl_operations")
        self.vl_operations.setContentsMargins(5, 25, 5, 25)
        self.btn_newSession = QPushButton(self.layoutWidget_base)
        self.btn_newSession.setObjectName(u"btn_newSession")

        self.vl_operations.addWidget(self.btn_newSession)

        self.btn_refreshSessions = QPushButton(self.layoutWidget_base)
        self.btn_refreshSessions.setObjectName(u"btn_refreshSessions")

        self.vl_operations.addWidget(self.btn_refreshSessions)

        self.cb_sessionList = QComboBox(self.layoutWidget_base)
        self.cb_sessionList.setObjectName(u"cb_sessionList")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cb_sessionList.sizePolicy().hasHeightForWidth())
        self.cb_sessionList.setSizePolicy(sizePolicy)
        self.cb_sessionList.setEditable(True)
        self.cb_sessionList.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.vl_operations.addWidget(self.cb_sessionList)

        self.btn_refreshOpertations = QPushButton(self.layoutWidget_base)
        self.btn_refreshOpertations.setObjectName(u"btn_refreshOpertations")

        self.vl_operations.addWidget(self.btn_refreshOpertations)

        self.cb_operationList = QComboBox(self.layoutWidget_base)
        self.cb_operationList.setObjectName(u"cb_operationList")
        sizePolicy.setHeightForWidth(self.cb_operationList.sizePolicy().hasHeightForWidth())
        self.cb_operationList.setSizePolicy(sizePolicy)
        self.cb_operationList.setEditable(True)
        self.cb_operationList.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.vl_operations.addWidget(self.cb_operationList)

        self.btn_newTagOperation = QPushButton(self.layoutWidget_base)
        self.btn_newTagOperation.setObjectName(u"btn_newTagOperation")

        self.vl_operations.addWidget(self.btn_newTagOperation)

        self.btn_deleteCurrentOperation = QPushButton(self.layoutWidget_base)
        self.btn_deleteCurrentOperation.setObjectName(u"btn_deleteCurrentOperation")

        self.vl_operations.addWidget(self.btn_deleteCurrentOperation)

        self.btn_resetTagOperation = QPushButton(self.layoutWidget_base)
        self.btn_resetTagOperation.setObjectName(u"btn_resetTagOperation")

        self.vl_operations.addWidget(self.btn_resetTagOperation)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.vl_operations.addItem(self.verticalSpacer)

        self.btn_getInitialPair = QPushButton(self.layoutWidget_base)
        self.btn_getInitialPair.setObjectName(u"btn_getInitialPair")

        self.vl_operations.addWidget(self.btn_getInitialPair)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.vl_operations.addItem(self.verticalSpacer_3)

        self.btn_showTaggedPointsTable = QPushButton(self.layoutWidget_base)
        self.btn_showTaggedPointsTable.setObjectName(u"btn_showTaggedPointsTable")
        self.btn_showTaggedPointsTable.setText(u"Show Tagged Points Table")

        self.vl_operations.addWidget(self.btn_showTaggedPointsTable)

        self.btn_sendTagFeatures = QPushButton(self.layoutWidget_base)
        self.btn_sendTagFeatures.setObjectName(u"btn_sendTagFeatures")

        self.vl_operations.addWidget(self.btn_sendTagFeatures)

        self.btn_mergeCurrentTagged = QPushButton(self.layoutWidget_base)
        self.btn_mergeCurrentTagged.setObjectName(u"btn_mergeCurrentTagged")

        self.vl_operations.addWidget(self.btn_mergeCurrentTagged)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.vl_operations.addItem(self.verticalSpacer_4)

        self.btn_kickoutCalculation = QPushButton(self.layoutWidget_base)
        self.btn_kickoutCalculation.setObjectName(u"btn_kickoutCalculation")
        self.btn_kickoutCalculation.setText(u"Kickout Calculation")

        self.vl_operations.addWidget(self.btn_kickoutCalculation)

        self.btn_reloadSFMResult = QPushButton(self.layoutWidget_base)
        self.btn_reloadSFMResult.setObjectName(u"btn_reloadSFMResult")

        self.vl_operations.addWidget(self.btn_reloadSFMResult)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.vl_operations.addItem(self.verticalSpacer_2)

        self.btn_close = QPushButton(self.layoutWidget_base)
        self.btn_close.setObjectName(u"btn_close")
        font = QFont()
        font.setPointSize(10)
        self.btn_close.setFont(font)
        self.btn_close.setText(u"Close")

        self.vl_operations.addWidget(self.btn_close)

        self.splitter_base.addWidget(self.layoutWidget_base)
        self.layoutWidget_scene = QWidget(self.splitter_base)
        self.layoutWidget_scene.setObjectName(u"layoutWidget_scene")
        self.vl_operationscenes = QVBoxLayout(self.layoutWidget_scene)
        self.vl_operationscenes.setObjectName(u"vl_operationscenes")
        self.vl_operationscenes.setContentsMargins(0, 0, 0, 15)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(self.layoutWidget_scene)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setText(u"Image Files:")

        self.horizontalLayout_3.addWidget(self.label_2)

        self.lE_imageFilesDir = QLineEdit(self.layoutWidget_scene)
        self.lE_imageFilesDir.setObjectName(u"lE_imageFilesDir")
        font1 = QFont()
        font1.setPointSize(12)
        self.lE_imageFilesDir.setFont(font1)
        self.lE_imageFilesDir.setText(u"")

        self.horizontalLayout_3.addWidget(self.lE_imageFilesDir)

        self.btn_browseImageDir = QPushButton(self.layoutWidget_scene)
        self.btn_browseImageDir.setObjectName(u"btn_browseImageDir")
        self.btn_browseImageDir.setText(u"Browse")

        self.horizontalLayout_3.addWidget(self.btn_browseImageDir)


        self.vl_operationscenes.addLayout(self.horizontalLayout_3)

        self.splitter_scenes = QSplitter(self.layoutWidget_scene)
        self.splitter_scenes.setObjectName(u"splitter_scenes")
        sizePolicy.setHeightForWidth(self.splitter_scenes.sizePolicy().hasHeightForWidth())
        self.splitter_scenes.setSizePolicy(sizePolicy)
        self.splitter_scenes.setOrientation(Qt.Horizontal)
        self.splitter_scenes.setOpaqueResize(True)
        self.splitter_scenes.setHandleWidth(8)
        self.splitter_scenes.setChildrenCollapsible(False)
        self.GB_ImageView = QGroupBox(self.splitter_scenes)
        self.GB_ImageView.setObjectName(u"GB_ImageView")
        self.GB_ImageView.setEnabled(True)
        sizePolicy.setHeightForWidth(self.GB_ImageView.sizePolicy().hasHeightForWidth())
        self.GB_ImageView.setSizePolicy(sizePolicy)
        self.GB_ImageView.setTitle(u"Select Image Points")
        self.GB_ImageView.setFlat(False)
        self.VB_imageView = QVBoxLayout(self.GB_ImageView)
        self.VB_imageView.setSpacing(0)
        self.VB_imageView.setObjectName(u"VB_imageView")
        self.VB_imageView.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.gridLayout.setContentsMargins(10, -1, -1, -1)
        self.label_4 = QLabel(self.GB_ImageView)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1, Qt.AlignTop)

        self.label_5 = QLabel(self.GB_ImageView)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout.addWidget(self.label_5, 0, 1, 1, 1)

        self.label_7 = QLabel(self.GB_ImageView)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 1, 0, 1, 1, Qt.AlignTop)

        self.label_6 = QLabel(self.GB_ImageView)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 1, 1, 1, 1, Qt.AlignTop)


        self.VB_imageView.addLayout(self.gridLayout)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setSpacing(10)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_3 = QLabel(self.GB_ImageView)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setText(u"Frame Offset:")

        self.horizontalLayout_6.addWidget(self.label_3)

        self.lE_FrameOffet = QLineEdit(self.GB_ImageView)
        self.lE_FrameOffet.setObjectName(u"lE_FrameOffet")
        self.lE_FrameOffet.setInputMask(u"")
        self.lE_FrameOffet.setText(u"0")

        self.horizontalLayout_6.addWidget(self.lE_FrameOffet)

        self.lb_imageFileName = QLabel(self.GB_ImageView)
        self.lb_imageFileName.setObjectName(u"lb_imageFileName")
        self.lb_imageFileName.setText(u"File Name:")

        self.horizontalLayout_6.addWidget(self.lb_imageFileName)

        self.lb_imagePosition = QLabel(self.GB_ImageView)
        self.lb_imagePosition.setObjectName(u"lb_imagePosition")
        self.lb_imagePosition.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_6.addWidget(self.lb_imagePosition)

        self.horizontalLayout_6.setStretch(0, 5)
        self.horizontalLayout_6.setStretch(1, 15)
        self.horizontalLayout_6.setStretch(2, 80)

        self.VB_imageView.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.vL_ImageView = QVBoxLayout()
        self.vL_ImageView.setObjectName(u"vL_ImageView")

        self.horizontalLayout_5.addLayout(self.vL_ImageView)


        self.VB_imageView.addLayout(self.horizontalLayout_5)

        self.VB_imageView.setStretch(0, 3)
        self.VB_imageView.setStretch(1, 3)
        self.VB_imageView.setStretch(2, 94)
        self.splitter_scenes.addWidget(self.GB_ImageView)
        self.GB_3DView = QGroupBox(self.splitter_scenes)
        self.GB_3DView.setObjectName(u"GB_3DView")
        self.GB_3DView.setTitle(u"Select 3D Points")
        self.GB_3DView.setFlat(False)
        self.VB_3DView = QVBoxLayout(self.GB_3DView)
        self.VB_3DView.setSpacing(0)
        self.VB_3DView.setObjectName(u"VB_3DView")
        self.VB_3DView.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(15)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(10, 5, -1, 5)
        self.btn_startSelectTagPoints = QPushButton(self.GB_3DView)
        self.btn_startSelectTagPoints.setObjectName(u"btn_startSelectTagPoints")
        self.btn_startSelectTagPoints.setText(u"  Selecting Mode  ")

        self.horizontalLayout_4.addWidget(self.btn_startSelectTagPoints)

        self.btn_endSelectTagPoints = QPushButton(self.GB_3DView)
        self.btn_endSelectTagPoints.setObjectName(u"btn_endSelectTagPoints")
        self.btn_endSelectTagPoints.setText(u"  End Selecting  ")

        self.horizontalLayout_4.addWidget(self.btn_endSelectTagPoints)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer)


        self.VB_3DView.addLayout(self.horizontalLayout_4)

        self.frm_3DView = QFrame(self.GB_3DView)
        self.frm_3DView.setObjectName(u"frm_3DView")
        self.frm_3DView.setFocusPolicy(Qt.TabFocus)
        self.frm_3DView.setFrameShape(QFrame.StyledPanel)
        self.frm_3DView.setFrameShadow(QFrame.Raised)
        self.vl_3DView = QVBoxLayout(self.frm_3DView)
        self.vl_3DView.setObjectName(u"vl_3DView")
        self.vl_3DView.setContentsMargins(0, 0, 0, 0)

        self.VB_3DView.addWidget(self.frm_3DView)

        self.VB_3DView.setStretch(0, 10)
        self.VB_3DView.setStretch(1, 90)
        self.splitter_scenes.addWidget(self.GB_3DView)

        self.vl_operationscenes.addWidget(self.splitter_scenes)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.hS_frame = QSlider(self.layoutWidget_scene)
        self.hS_frame.setObjectName(u"hS_frame")
        self.hS_frame.setOrientation(Qt.Horizontal)

        self.horizontalLayout_2.addWidget(self.hS_frame)

        self.btn_previousFrame = QPushButton(self.layoutWidget_scene)
        self.btn_previousFrame.setObjectName(u"btn_previousFrame")
        self.btn_previousFrame.setText(u"Previous")

        self.horizontalLayout_2.addWidget(self.btn_previousFrame)

        self.btn_nextFrame = QPushButton(self.layoutWidget_scene)
        self.btn_nextFrame.setObjectName(u"btn_nextFrame")
        self.btn_nextFrame.setText(u"Next")

        self.horizontalLayout_2.addWidget(self.btn_nextFrame)

        self.label = QLabel(self.layoutWidget_scene)
        self.label.setObjectName(u"label")
        self.label.setText(u"Frame:")

        self.horizontalLayout_2.addWidget(self.label)

        self.sB_frame = QSpinBox(self.layoutWidget_scene)
        self.sB_frame.setObjectName(u"sB_frame")

        self.horizontalLayout_2.addWidget(self.sB_frame)

        self.horizontalLayout_2.setStretch(0, 80)
        self.horizontalLayout_2.setStretch(1, 5)
        self.horizontalLayout_2.setStretch(2, 5)
        self.horizontalLayout_2.setStretch(3, 5)
        self.horizontalLayout_2.setStretch(4, 10)

        self.vl_operationscenes.addLayout(self.horizontalLayout_2)

        self.vl_operationscenes.setStretch(0, 5)
        self.vl_operationscenes.setStretch(1, 80)
        self.vl_operationscenes.setStretch(2, 5)
        self.splitter_base.addWidget(self.layoutWidget_scene)

        self.vl_base.addWidget(self.splitter_base)


        self.retranslateUi(SelectCorrespondences)

        QMetaObject.connectSlotsByName(SelectCorrespondences)
    # setupUi

    def retranslateUi(self, SelectCorrespondences):
        self.btn_newSession.setText(QCoreApplication.translate("SelectCorrespondences", u"New Session", None))
        self.btn_refreshSessions.setText(QCoreApplication.translate("SelectCorrespondences", u"Refesh Sessions", None))
        self.btn_refreshOpertations.setText(QCoreApplication.translate("SelectCorrespondences", u"Refresh Operations", None))
        self.btn_newTagOperation.setText(QCoreApplication.translate("SelectCorrespondences", u"New Tag Operation", None))
        self.btn_deleteCurrentOperation.setText(QCoreApplication.translate("SelectCorrespondences", u"Delete Current Operation", None))
        self.btn_resetTagOperation.setText(QCoreApplication.translate("SelectCorrespondences", u"Reset Current Tag Operation", None))
        self.btn_getInitialPair.setText(QCoreApplication.translate("SelectCorrespondences", u"Get Initial Pair", None))
        self.btn_sendTagFeatures.setText(QCoreApplication.translate("SelectCorrespondences", u"Send current tagged", None))
        self.btn_mergeCurrentTagged.setText(QCoreApplication.translate("SelectCorrespondences", u"Merge Current Tagged", None))
        self.btn_reloadSFMResult.setText(QCoreApplication.translate("SelectCorrespondences", u"Reload SFM Result", None))
        self.label_4.setText(QCoreApplication.translate("SelectCorrespondences", u"Shift + LMB:  Add new point tag", None))
        self.label_5.setText(QCoreApplication.translate("SelectCorrespondences", u"Shift + RMB:  Remove selected tag", None))
        self.label_7.setText(QCoreApplication.translate("SelectCorrespondences", u"Mouse Wheel:  Zoom view", None))
        self.label_6.setText(QCoreApplication.translate("SelectCorrespondences", u"Alt + LMB:  Pane view          ", None))
        self.lb_imagePosition.setText(QCoreApplication.translate("SelectCorrespondences", u"x: y: ", None))
        pass
    # retranslateUi

