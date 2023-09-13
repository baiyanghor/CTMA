import maya.OpenMayaUI as OMUI
import maya.api.OpenMaya as OM2
import maya.cmds as MCMDS
from PySide2.QtCore import Qt, QObject
from PySide2.QtWidgets import QWidget, QMainWindow, QDesktopWidget, QFileDialog, QDialog
from PySide2.QtGui import QIntValidator
from shiboken2 import wrapInstance
import os
import sys
import GraphicsViewWithTag
reload(GraphicsViewWithTag)

from UIDesign.SelectCorrespondences import SelectCorrespondences
reload(SelectCorrespondences)
from UIDesign.SelectCorrespondences import CorrespondingPointsTable
reload(CorrespondingPointsTable)
import SFMCalculateInterface
reload(SFMCalculateInterface)

from Configure import ConfigureOperators

import SessionManagement
reload(SessionManagement)

mayaMainWindowPtr = OMUI.MQtUtil.mainWindow()
mayaMainWindow = wrapInstance(long(mayaMainWindowPtr), QMainWindow)


class SelectCorrespondPonitsDLG(QWidget):
    toolName = "SelectCorrespondencePoints"
    DLG_STATUS_IDLE = 0
    DLG_STATUS_IMAGELOADED = 1
    DLG_STATUS_TAGGING = 2
    DLG_STATUS_RECALCULATING = 3
    
    def __init__(self, parent=None, databaseHandler=None, dataModelHandler=None, local_workdir=''):
        self.callbackID = None
        self.view_click_connencted = False
        #self.deleteInstances()
        self.deleteWidgetUI(self.toolName)
        super(SelectCorrespondPonitsDLG, self).__init__(parent=parent)
        self.ui = None
        self.modelPointsView = None
        self.QtObj_3DView = None
        self.imageQtScene = None
        self.imageQtView = None        
        self.imageFilesDir = ''
        self.currentFrame = 0
        self.startFrame = 0
        self._frameCount = 0
        self.imageFileNameList = []
        self.zoomInFactor = 1.15
        self.zoomOutFactor = 1 / self.zoomInFactor
        self.topItem = None
        self._DatabaseManagerHandler = databaseHandler
        self._dataModelHandler = dataModelHandler
        self.currentFileDir = local_workdir
        self.configHandler = ConfigureOperators.ConfigureOps(self.currentFileDir)
        
        self.taggedTableDLG = CorrespondingPointsTableDLG(self, self.configHandler)
        self.mayaSceneModel = MayaSceneDataModel()
        
        self.taggedTableDLG.initialTable(self._dataModelHandler)
        self.mayaSceneModel.setDatabaseModel(self._dataModelHandler)
        self._dataModelReady = False
        
        self._sfmSessionID = ''
        self._operationID = 0
        self._focalLength = 933.3352
        # 'Height,Width' with no space in there 
        self._imageHeightWidthString = ''
        
        self.setWindowFlags(Qt.Window)
        self.initUIComponents()
        self.setWindowTitle("Select Correspondences")

        self.setObjectName(self.__class__.toolName)
        self.setAttribute(Qt.WA_DeleteOnClose)


        
        self._sfm_ops = SFMCalculateInterface.SFMCalculatorInterface(self.configHandler.getSFMServiceDir())
        
        self.connectSlots()
        self.DLG_Status = self.DLG_STATUS_IDLE
        self.switchDLGStatus(self.DLG_STATUS_IDLE)
        self.ui.splitter_base.setSizes([300, 1600])
        self.ui.splitter_scenes.setSizes([800,800])
        
        self.initialModelView()
        
        self.setGeometry(0, 0, 1200, 1000)        
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRect = self.frameGeometry()
        qtRect.moveCenter(centerPoint)
        self.move(qtRect.topLeft())

    def initUIComponents(self):
        self.ui = SelectCorrespondences.Ui_SelectCorrespondences()
        self.ui.setupUi(self.window())
        self.imageQtView = GraphicsViewWithTag.GraphicsViewWithTag(self)        
        self.ui.vL_ImageView.addWidget(self.imageQtView)        
        self.ui.lE_FrameOffet.setValidator(QIntValidator())
        self.ui.cb_sessionList.lineEdit().setAlignment(Qt.AlignRight)
        self.ui.cb_operationList.lineEdit().setAlignment(Qt.AlignRight)
        self.ui.cb_sessionList.lineEdit().setReadOnly(True)
        self.ui.cb_operationList.lineEdit().setReadOnly(True)
        self.refreshUISessionList()
        self.refreshUIOperationList(self._sfmSessionID)
    
    def refreshUISessionList(self):
        session_list = self._DatabaseManagerHandler.getSessions()
        if len(session_list) > 0:
            self.ui.cb_sessionList.clear()
            self.ui.cb_sessionList.insertItems(0, session_list)
            self._sfmSessionID = self.ui.cb_sessionList.currentText()
    
    def refreshUIOperationList(self, session_id):
        operation_list = self._DatabaseManagerHandler.getOperations(self._sfmSessionID)
        if len(operation_list) > 0:
            self.ui.cb_operationList.clear()
            self.ui.cb_operationList.insertItems(0, operation_list)
            self._operationID = int(self.ui.cb_operationList.currentText().split(':')[0])
            self._dataModelReady = self._dataModelHandler.refreshDataQuery(self._sfmSessionID, self._operationID)


    def initialModelView(self):
        MCMDS.setParent('vl_3DView')
        modelPointsView = MCMDS.paneLayout()
        
        if MCMDS.modelEditor("TagCorresponding", query=True, exists=True):
            MCMDS.deleteUI("TagCorresponding")
            
        MCMDS.modelEditor("TagCorresponding", camera = 'persp',
                            displayAppearance = 'flatShaded',
                            lights = True,
                            displayLights = 'all',
                            wireframeOnShaded = True)

        qtPtr = OMUI.MQtUtil.findControl(modelPointsView) 
        self.QtObj_3DView = wrapInstance(long(qtPtr), QWidget)
        self.ui.vl_3DView.addWidget(self.QtObj_3DView)

    def initialImageView(self, imageSceneModel):
        self.imageQtScene = imageSceneModel
        self.imageQtView.setScene(imageSceneModel)
        self.imageQtScene.disableTag()        

    def __ToQtObject(self, mayaName):
        ptr = OMUI.MQtUtil.findControl(mayaName)
        if ptr is None:
            ptr = OMUI.MQtUtil.findLayout(mayaName)
            if ptr is None:
                ptr = OMUI.MQtUtil.findMenuItem(mayaName)
    
        if ptr is not None:
            return wrapInstance(long(ptr), QWidget)

        
    def connectSlots(self):        
        self.ui.btn_close.clicked.connect(self.onClose)
        self.ui.btn_browseImageDir.clicked.connect(self.onSelectImageFilePath)
        self.ui.hS_frame.valueChanged.connect(self.onHSliderFrameValueChanged)
        self.ui.btn_nextFrame.clicked.connect(self.onNextFrame)
        self.ui.btn_previousFrame.clicked.connect(self.onPreviousFrame)
        self.ui.sB_frame.valueChanged.connect(self.onSBFrameChanged)
        self.ui.lE_imageFilesDir.textChanged.connect(self.onImagePathChange)
        self.ui.btn_showTaggedPointsTable.clicked.connect(self.onShowTaggedPointsTable)
        self.ui.btn_startSelectTagPoints.clicked.connect(self.onStartDefineScenePoints)
        self.ui.btn_endSelectTagPoints.clicked.connect(self.onEndDefineScenePoints)
        self.ui.btn_getInitialPair.clicked.connect(self.onGetInitialSFMImagePair)
        self.ui.btn_kickoutCalculation.clicked.connect(self.onKickoutSFMCalc)
        self.ui.btn_newSession.clicked.connect(self.onNewSession)
        self.ui.cb_sessionList.currentIndexChanged.connect(self.onSessionListChanged)
        self.ui.cb_operationList.currentIndexChanged.connect(self.onOperationListChanged)
        self.ui.btn_newTagOperation.clicked.connect(self.onNewTagOperation)
        self.ui.btn_refreshOpertations.clicked.connect(self.onRefreshOperationList)
        self.ui.btn_refreshSessions.clicked.connect(self.onRefreshSessionList)

    def switchDLGStatus(self, newStatus):
        if newStatus == self.DLG_STATUS_IMAGELOADED:
            if self.DLG_Status == self.DLG_STATUS_TAGGING:
                self.disconnectSceneSelectionEvent()
                MCMDS.selectMode(object = True)
                MCMDS.select(clear = True)
            else:
                self.ui.hS_frame.setEnabled(True)
                self.ui.sB_frame.setEnabled(True)
                self.ui.btn_nextFrame.setEnabled(True)
                self.ui.btn_previousFrame.setEnabled(True)
                self.ui.btn_getInitialPair.setEnabled(True)
                self.ui.hS_frame.setMinimum(self.startFrame)
                self.ui.hS_frame.setMaximum(self.startFrame + self._frameCount - 1)
                self.ui.sB_frame.setMinimum(self.startFrame)
                self.ui.sB_frame.setMaximum(self.startFrame + self._frameCount - 1)
            self.ui.btn_startSelectTagPoints.setEnabled(True)
            self.ui.btn_endSelectTagPoints.setEnabled(False)
            self.ui.btn_newSession.setEnabled(True)
            
        elif newStatus == self.DLG_STATUS_IDLE:
            self.ui.hS_frame.setEnabled(False)
            self.ui.sB_frame.setEnabled(False)
            self.ui.btn_nextFrame.setEnabled(False)
            self.ui.btn_showTaggedPointsTable.setEnabled(True)
            self.ui.btn_getInitialPair.setEnabled(False)
            self.ui.btn_kickoutCalculation.setEnabled(False)
            self.ui.btn_previousFrame.setEnabled(False)
            self.ui.btn_startSelectTagPoints.setEnabled(False)
            self.ui.btn_endSelectTagPoints.setEnabled(False)
            self.ui.btn_browseImageDir.setEnabled(True)
            self.ui.btn_newSession.setEnabled(False)
            self.disconnectSceneSelectionEvent()
                    
        elif newStatus == self.DLG_STATUS_TAGGING:
            self.connectSceneSelectionEvent()
            self.ui.btn_startSelectTagPoints.setEnabled(False)
            self.ui.btn_endSelectTagPoints.setEnabled(True)
            self.ui.btn_showTaggedPointsTable.setEnabled(True)
            self.ui.btn_getInitialPair.setEnabled(True)
            self.ui.btn_kickoutCalculation.setEnabled(True)         
            MCMDS.selectMode(component = True)
            MCMDS.select(clear = True)
            
        elif newStatus == self.DLG_STATUS_RECALCULATING:
            self.ui.hS_frame.setEnabled(False)
            self.ui.sB_frame.setEnabled(False)
            self.ui.btn_nextFrame.setEnabled(False)
            self.ui.btn_showTaggedPointsTable.setEnabled(False)
            self.ui.btn_getInitialPair.setEnabled(False)
            self.ui.btn_kickoutCalculation.setEnabled(False)
            self.ui.btn_previousFrame.setEnabled(False)
            self.ui.btn_startSelectTagPoints.setEnabled(False)
            self.ui.btn_endSelectTagPoints.setEnabled(False)
            self.ui.btn_browseImageDir.setEnabled(False)
            self.disconnectSceneSelectionEvent()            

        self.DLG_Status = newStatus
        
    def onClose(self):
        self._DatabaseManagerHandler.closeDatabase()
        self.disconnectSceneSelectionEvent()
        self.taggedTableDLG.close()
        #self.deleteInstances()    
        self.close()

    def hideEvent(self, event):
        self._DatabaseManagerHandler.closeDatabase()
        self.disconnectSceneSelectionEvent()
        self.taggedTableDLG.close()
        #self.deleteInstances()

    def onSelectImageFilePath(self):
        imageFileDig = QFileDialog(self)
        imageFileDig.setNameFilter("Images (*.png *.xpm *.jpg)")
        imageFilePath = imageFileDig.getExistingDirectory(self, 'Select Image Sequence Path', self.currentFileDir)
        self.ui.lE_imageFilesDir.setText(imageFilePath)

    def onHSliderFrameValueChanged(self, value):
        self.ui.sB_frame.setValue(value)
        self.showImageFrame(value)

    def onImageWheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoomFactor = self.zoomInFactor
        else:
            zoomFactor = self.zoomOutFactor
        self.ui.gV_imageView.scale(zoomFactor, zoomFactor)

    def onNextFrame(self):
        if self.currentFrame < self._frameCount - 1:
            self.showImageFrame(self.currentFrame + 1)
    
    def onPreviousFrame(self):
        if self.currentFrame > 0:
            self.showImageFrame(self.currentFrame - 1)
    
    def onSBFrameChanged(self, value):
        if self.DLG_Status == self.DLG_STATUS_IMAGELOADED:
            if value in range(0, self._frameCount):
                self.ui.hS_frame.setValue(value)
                self.showImageFrame(value)        

    def onImagePathChange(self, text):
        if os.path.isdir(text):
            self.imageFilesDir = text.strip()
            self.LoadImageListToView()
            self.showImageFrame(0)
            self.ui.lE_imageFilesDir.clearFocus()            

    def onShowTaggedPointsTable(self):
        self.taggedTableDLG.set_session_id(self._sfmSessionID)
        self.taggedTableDLG.set_operation_id(self._operationID)
        self.taggedTableDLG.setWindowTitle(self._sfmSessionID)
        self.taggedTableDLG.show()

        
    def onStartDefineScenePoints(self):
        self.switchDLGStatus(self.DLG_STATUS_TAGGING)
    
    def onEndDefineScenePoints(self):
        self.switchDLGStatus(self.DLG_STATUS_IMAGELOADED)
        
    def onGetInitialSFMImagePair(self):
        if self.DLG_Status == self.DLG_STATUS_IMAGELOADED or self.DLG_STATUS_TAGGING:                
            sfm_initial_pair = self._sfm_ops.get_initial_sfm_pair(self._sfmSessionID)
            if len(sfm_initial_pair) < 3:
                print("Server side not ready yet!")
                return None
            else:
                print("Get initial pair succeeded!")
                print(sfm_initial_pair)
                locat_frame = int(sfm_initial_pair.split(',')[0])
                self.showImageFrame(locat_frame)
            
    def onKickoutSFMCalc(self):
        user_image_data = self._DatabaseManagerHandler.get_user_image_ponits()
        user_world_data = self._DatabaseManagerHandler.get_user_world_points()
        print "user_image_data"
        print user_image_data
        print "user_world_data"
        print user_world_data
        
    def onNewSession(self):
        if self.DLG_Status == self.DLG_STATUS_IMAGELOADED or self.DLG_STATUS_TAGGING:
            self._sfmSessionID = self._sfm_ops.new_session(os.getenv('USERNAME'), '_')
            if self._sfmSessionID == 'Failure':
                print('Create SFM session failure, maybe something wrong with SFM server side.')
                self._sfmSessionID = ''
                return
            # Get ready to tagged table to right to calculator
            self.taggedTableDLG.set_session_id(self._sfmSessionID)
                
            if self._sfm_ops.set_global_sfm_info(self._sfmSessionID, self._focalLength, self._frameCount,
                                                 self._imageHeightWidthString, self.imageFilesDir):
                print("Set image path succeeded!")
            else:
                print("Set image path server side not ready yet!")
            
            # Insert session ID in UI
            self._DatabaseManagerHandler.newSession(self._sfmSessionID)
            self.refreshUISessionList()
            self.ui.cb_sessionList.setCurrentText(self._sfmSessionID)
    
    def onRefreshSessionList(self):
        self.refreshUISessionList()
        self.ui.cb_sessionList.setCurrentText(self._sfmSessionID)
        
    def onSessionListChanged(self, index):
        self._sfmSessionID = self.ui.cb_sessionList.itemText(index)
        self.onRefreshOperationList()
        self._dataModelReady = self._dataModelHandler.refreshDataQuery(self._sfmSessionID, self._operationID)
        
    def onOperationListChanged(self, index):
        operation_text = self.ui.cb_operationList.itemText(index)
        if len(operation_text) > 0:
            self._operationID = operation_text.split(':')[0]
            self._dataModelReady = self._dataModelHandler.refreshDataQuery(self._sfmSessionID, self._operationID)
        else:
            self._operationID = 0
            self._dataModelReady = False
        print "Debug: Current Operation ID {0}".format(self._operationID)
        
    def onNewTagOperation(self):
        self._operationID = self._DatabaseManagerHandler.newTagOperation(self._sfmSessionID)
        self._dataModelReady = self._dataModelHandler.refreshDataQuery(self._sfmSessionID, self._operationID)
        self.imageQtScene.enableTag()
        self.refreshUIOperationList(self._sfmSessionID)
    
    def onRefreshOperationList(self):
        self.refreshUIOperationList(self._sfmSessionID)
        op_str = self.ui.cb_operationList.currentText().split(':')[0]
        if op_str:
            self._operationID = int(op_str)
        else:
            self._operationID = 0
    
    def LoadImageListToView(self):
        self.imageFileNameList[:] = []
        self.imageQtScene.setPosIndicatorHandler(self.ui.lb_imagePosition)
        self.imageQtScene.reinitScene()

        if os.path.isdir(self.imageFilesDir):

            validFileTpyes = self.configHandler.getValidImageType()
            for aFile in sorted(os.listdir(self.imageFilesDir)):
                if aFile.split('.')[-1] in validFileTpyes:
                    self.imageFileNameList.append(aFile)

            self.imageFileNameList = sorted(self.imageFileNameList)
            image_count, image_size_str = self.imageQtScene.loadImageList(self.imageFilesDir, self.imageFileNameList)
            self._imageHeightWidthString = image_size_str
            self._frameCount = image_count
            self.switchDLGStatus(self.DLG_STATUS_IMAGELOADED)

    
    def getOffsetedFrameWithImageName(self, listIndex = 0):
        return listIndex + int(self.ui.lE_FrameOffet.text())

    
    def showImageFrame(self, frame = 0):
        if len(self.imageQtScene.items()) > 0:
            self.imageQtScene.showFrame(frame)
            self.currentFrame = frame
            self.setCurrentFrameUI()     

    def setCurrentFrameUI(self):
        self.ui.hS_frame.setValue(self.currentFrame)
        self.ui.sB_frame.setValue(self.currentFrame)
        self.ui.lb_imageFileName.setText("File Name:  " + self.imageFileNameList[self.currentFrame])

    def deleteControl(self, control):
        if MCMDS.workspaceControl(control, q=True, exists=True):
            MCMDS.workspaceControl(control, e=True, close=True)
            MCMDS.deleteUI(control, control=True)


    #def deleteInstances(self):
        #for obj in mayaMainWindow.children():            
            #if obj.objectName() == self.__class__.toolName:
                #mayaMainWindow.removeDockWidget(obj)
                #obj.setParent(None)
                #obj.deleteLater()

    def setDatabaseManagerHandler(self, databaseManagerHandler):
        self._DatabaseManagerHandler = databaseManagerHandler
        
    
    def connectSceneSelectionEvent(self):
        self.callbackID = OM2.MEventMessage.addEventCallback('SelectionChanged', self.onViewportPointSelection)
        self.view_click_connencted = True
    
    def disconnectSceneSelectionEvent(self):
        if self.view_click_connencted:
            OM2.MMessage.removeCallback(self.callbackID)
            self.view_click_connencted = False

    def onViewportPointSelection(self, *args):
        self.mayaSceneModel.onViewportClicked(self.currentFrame + 1)

    def deleteWidgetUI(self, ui_name):
        for obj in mayaMainWindow.children():
            if type(obj) == QWidget:
                if obj.objectName() == ui_name:
                    MCMDS.deleteUI(obj)


class CorrespondingPointsTableDLG(QDialog):
    def __init__(self, parent = None, configHandler=None):
        super(CorrespondingPointsTableDLG, self).__init__(parent = parent)
        self.ui = None
        self._DataTableModel = None
        self._sfmSessionID = ''
        self._operationID = 0
        self._configHandler = configHandler
        self.setWindowFlags(Qt.Window|Qt.WindowStaysOnTopHint)
        self.initialUIComponents()
        self.resize(1000, 600)
        self.connectSlots()
        
    def set_session_id(self, in_session_id):
        self._sfmSessionID = in_session_id
    
    def set_operation_id(self, operation_id):
        self._operationID = operation_id

    def initialUIComponents(self):
        self.ui = CorrespondingPointsTable.Ui_dlg_taggedPointsTable()
        self.ui.setupUi(self)
        self.ui.pb_operating.setMinimum(0)
        self.ui.pb_operating.setMaximum(100)
    
    def connectSlots(self):
        self.ui.btn_close.clicked.connect(self.onClose)
        self.ui.btn_sendToCalculator.clicked.connect(self.onSendToCalculator)
        
    def onClose(self):
        self.hide()
    
    def onSendToCalculator(self):
        if self._DataTableModel is None:
            return
        
        if self._sfmSessionID == '':
            return
        self._DataTableModel.refreshDataQuery(self._sfmSessionID, self._operationID)
        send_ret = self._DataTableModel.sendTaggedPoints()
        sfm_service_work_dir = self._configHandler.getSFMServiceDir()
        sfm_ops = SFMCalculateInterface.SFMCalculatorInterface(sfm_service_work_dir)
        if len(self._sfmSessionID) > 0 and self._operationID > 0:
            sfm_ops.set_user_tagged_done(self._sfmSessionID, str(self._operationID))
        else:
            print "Require operation ID and session ID"
        

    def initialTable(self, dataTableModel):
        self._DataTableModel = dataTableModel
        self.ui.tV_taggedPoints.setModel(self._DataTableModel.toViewModel)
        self.ui.tV_taggedPoints.setColumnHidden(0, True)        
        self.ui.tV_taggedPoints.setSortingEnabled(True)

    def resortView(self):
        self._DataTableModel.refeshDataModel()
        self.ui.tV_taggedPoints.sortByColumn(1, Qt.AscendingOrder)
        self.ui.tV_taggedPoints.sortByColumn(3, Qt.AscendingOrder)

    def showEvent(self, event):
        self._DataTableModel.refreshViewDataModel()
        self.ui.pb_operating.setVisible(False)

        
class MayaSceneDataModel(QObject):
    def __init__(self, parent=None):
        super(MayaSceneDataModel, self).__init__(parent=parent)
        self.taggedPointsGroup = 'TAGGED_POINTS'
        self.annotationsGroup = 'TRACKTOOL_ANNOTATIONS'
        # Dict structure {'Tag Text': {'fullPointID': 'full Point ID', 'annotateName': 'annotation node name', position: [x, y, z]}}
        self.taggedScenePointsList = []
        self._DatabaseModel = None

        if MCMDS.objExists(self.annotationsGroup):
            MCMDS.delete(self.annotationsGroup)
            
    def clearAnnotations(self):
        if MCMDS.objExists(self.annotationsGroup):
            MCMDS.delete(self.annotationsGroup)        

    def setDatabaseModel(self, dataModel):
        self._DatabaseModel = dataModel

    def onViewportClicked(self, FrameNumber):
        meshName = None
        pointNumber = None
        pointPosition = None
        sl_list = OM2.MGlobal.getActiveSelectionList()

        if sl_list.length() > 0:
            (dp, obj) = sl_list.getComponent(0)

            if obj.apiType() == OM2.MFn.kInvalid:
                return

            else:
                meshName = dp.partialPathName()
                vertex_iter = OM2.MItMeshVertex(dp, obj)

                while not vertex_iter.isDone():
                    pointNumber = vertex_iter.index()
                    aPoint = vertex_iter.position(OM2.MSpace.kWorld)
                    pointPosition = [aPoint.x, aPoint.y, aPoint.z]
                    vertex_iter.next()

                fullPointID = '.'.join([meshName, str(pointNumber)])

                if self.pointExists(fullPointID):
                    self.removeScenePointInView(fullPointID)
                    indexToRemove = self.removeScenePoint(fullPointID)
                    self._DatabaseModel.removePointsInScene(indexToRemove)
                    self.resortTagPointsInView()
                else:                
                    tagObjectName = self.addScenePointInView(meshName, pointPosition)
                    self.rememberScenePoint(fullPointID, tagObjectName, pointPosition)
                    self._DatabaseModel.insertScenePoint(len(self.taggedScenePointsList),
                                                            pointPosition[0],
                                                            pointPosition[1],
                                                            pointPosition[2],
                                                            )

            MCMDS.select(clear = True)
            MCMDS.select(meshName, replace = True)
            MCMDS.selectMode(component = True)                
        
    def resortTagPointsInView(self):
        for index, tagItem in enumerate(self.taggedScenePointsList):
            anoName = MCMDS.listRelatives(tagItem['annotationName'])[0]
            MCMDS.setAttr(anoName + '.text', str(index + 1), type='string')
    

    def removeScenePointInView(self, fullPointID = ''):
        tagObjectName = ''           
        for tagItem in self.taggedScenePointsList:
            if fullPointID == tagItem['fullPointID']:
                tagObjectName = tagItem['annotationName']
                break
        
        if MCMDS.objExists(tagObjectName):
            MCMDS.delete(tagObjectName)
    
    def rememberScenePoint(self, fullPointID = '', tagObjectName = '', pointPosition = None):
        if fullPointID == '' or tagObjectName == '': return
        dataDict = {'fullPointID': fullPointID, 'annotationName': tagObjectName, 'position': pointPosition}
        self.taggedScenePointsList.append(dataDict)
        
    
    def removeScenePoint(self, fullPointID = ''):
        if fullPointID == '': return

        for tagItem in self.taggedScenePointsList:
            if fullPointID == tagItem['fullPointID']:
                index = self.taggedScenePointsList.index(tagItem)
                self.taggedScenePointsList.pop(index)
                return index
            
        return None    
    
    def pointExists(self, fullPointID):                
        for tagSceneItem in self.taggedScenePointsList:
            existPointID = tagSceneItem['fullPointID']
            if fullPointID == existPointID:
                return True
        return False                  
    
    def addScenePointInView(self, meshName, pointPosition):
        if not MCMDS.objExists(self.annotationsGroup):
            MCMDS.group(empty = True, name = self.annotationsGroup)
            
        pointTagShape = MCMDS.annotate(meshName, text=str(self.getNewScenePointIndex()), point=pointPosition)
        MCMDS.setAttr(pointTagShape + '.displayArrow', 0)

        if not MCMDS.objExists(self.taggedPointsGroup):
            MCMDS.group(empty=True, name=self.taggedPointsGroup)
            MCMDS.parent(self.taggedPointsGroup, self.annotationsGroup)

        pointTag = MCMDS.listRelatives(pointTagShape, parent = True)[0]
            
        MCMDS.parent(pointTag, self.taggedPointsGroup)
        
        return pointTag

    def getNewScenePointIndex(self):
        return len(self.taggedScenePointsList) + 1

