from PySide2.QtCore import QObject
from PySide2.QtWidgets import QMainWindow, QWidget
import maya.OpenMayaUI as OMUI
import maya.cmds as MCMDS

from shiboken2 import wrapInstance

import View
reload(View)

import Model
reload(Model)

mayaMainWindowPtr = OMUI.MQtUtil.mainWindow()
mayaMainWindow = wrapInstance(long(mayaMainWindowPtr), QMainWindow)


class MayaInterfaceController(QObject):
    def __init__(self, parent=None, localWorkDir=''):
        super(MayaInterfaceController, self).__init__(parent=parent)
        self.setObjectName("SelectCorrespondingControl")
        
        self.databaseManager = Model.DatabaseManager(self, localWorkDir)
        databaseCon = self.databaseManager.getDefaultConnection()
        if databaseCon is None:
            print("Database initialize failure!")
            return None
        viewToolName = "SelectCorrespondencePoints"
        
        self.deleteWidgetUI(viewToolName)
        
        self.databaseManager.initialTables()
        
        self.dataTableModel = Model.SelectedPointsTableModel(self, databaseCon, localWorkDir)
        
        self.imageModel = Model.GraphicsSceneWithTag(self)
        
        self.imageModel.setTagDataTableModel(self.dataTableModel)   
        
        self.view = View.SelectCorrespondPonitsDLG(mayaMainWindow, self.databaseManager, self.dataTableModel, localWorkDir)
        
        self.view.initialImageView(self.imageModel)
       
    def deleteWidgetUI(self, ui_name):
        for obj in mayaMainWindow.children():
            if type(obj) == QWidget:
                if obj.objectName() == ui_name:
                    MCMDS.deleteUI(obj)
    
    def exe(self):
        if self.view:
            self.view.show()
