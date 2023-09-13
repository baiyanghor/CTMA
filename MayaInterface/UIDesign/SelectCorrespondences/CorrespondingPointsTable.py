# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'CorrespondingPointsTable.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_dlg_taggedPointsTable(object):
    def setupUi(self, dlg_taggedPointsTable):
        if not dlg_taggedPointsTable.objectName():
            dlg_taggedPointsTable.setObjectName(u"dlg_taggedPointsTable")
        dlg_taggedPointsTable.resize(868, 659)
        self.verticalLayout = QVBoxLayout(dlg_taggedPointsTable)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.tV_taggedPoints = QTableView(dlg_taggedPointsTable)
        self.tV_taggedPoints.setObjectName(u"tV_taggedPoints")

        self.horizontalLayout.addWidget(self.tV_taggedPoints)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(15)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.btn_sendToCalculator = QPushButton(dlg_taggedPointsTable)
        self.btn_sendToCalculator.setObjectName(u"btn_sendToCalculator")
        font = QFont()
        font.setFamily(u"Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.btn_sendToCalculator.setFont(font)
        self.btn_sendToCalculator.setText(u"Send to Calculator")

        self.horizontalLayout_2.addWidget(self.btn_sendToCalculator)

        self.btn_close = QPushButton(dlg_taggedPointsTable)
        self.btn_close.setObjectName(u"btn_close")
        font1 = QFont()
        font1.setFamily(u"Arial")
        font1.setPointSize(10)
        self.btn_close.setFont(font1)
        self.btn_close.setText(u"Close")

        self.horizontalLayout_2.addWidget(self.btn_close)

        self.horizontalLayout_2.setStretch(0, 50)
        self.horizontalLayout_2.setStretch(1, 25)
        self.horizontalLayout_2.setStretch(2, 25)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.pb_operating = QProgressBar(dlg_taggedPointsTable)
        self.pb_operating.setObjectName(u"pb_operating")
        font2 = QFont()
        font2.setPointSize(2)
        self.pb_operating.setFont(font2)
        self.pb_operating.setValue(0)
        self.pb_operating.setTextVisible(False)
        self.pb_operating.setFormat(u"%p%")

        self.verticalLayout.addWidget(self.pb_operating)

        self.verticalLayout.setStretch(0, 90)
        self.verticalLayout.setStretch(1, 5)
        self.verticalLayout.setStretch(2, 5)

        self.retranslateUi(dlg_taggedPointsTable)

        QMetaObject.connectSlotsByName(dlg_taggedPointsTable)
    # setupUi

    def retranslateUi(self, dlg_taggedPointsTable):
        dlg_taggedPointsTable.setWindowTitle(QCoreApplication.translate("dlg_taggedPointsTable", u"Form", None))
    # retranslateUi

