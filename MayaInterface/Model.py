from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtSql import QSqlDatabase, QSqlTableModel, QSqlQuery, QSqlQueryModel
import shutil
from json import load, dump
from Configure import ConfigureOperators
reload(ConfigureOperators)
import os

from SessionManagement import OPERATION


class SESSION_TYPE():
    NEW = 0
    EXISTING = 1

class TagPoint(QGraphicsRectItem):
    def __init__(self, rect = None, parent = None):
        super(TagPoint, self).__init__(rect, parent)
        self.tag_line_width = 2
        self.hightPen = QPen(QBrush(QColor(Qt.red)), self.tag_line_width)
        self.tag_pen = QPen(QBrush(QColor(Qt.green)), self.tag_line_width)
        self.setPen(self.tag_pen)


class ImageSceneItem(QGraphicsPixmapItem):
    def __init__(self, pixmap, frameNumber, fileName, parent = None):
        super(ImageSceneItem, self).__init__(pixmap, parent = parent)
        self.tag_rect_size = 5
        self.fileName = fileName
        self.frame_Number = frameNumber
        self.fontSize = 10
        self._posIndicatorHandler = None
        self._tagEnabled = False
        
    def enableTag(self):
        self._tagEnabled = True
        
    def disableTag(self):
        self._tagEnabled = False
        
    def setPosIndicatorHandler(self, pos_label):
        self._posIndicatorHandler = pos_label
    
    def setTagDataTableModel(self, dataTableModel):
        self._TagDataTableModel = dataTableModel

    def getNewIndex(self):        
        return len(self.childItems())

    def mousePressEvent(self, event):
        if self._tagEnabled:
            if event.modifiers() == Qt.ShiftModifier and event.button() == Qt.LeftButton:
                self.addPointTag(event.scenePos())
        QGraphicsPixmapItem.mousePressEvent(self, event)
            
    def hoverMoveEvent(self, event):
        pos = event.pos()
        label_text = "x: {0} y: {1}".format(round(pos.x(), 3), round(pos.y(), 3))
        self._posIndicatorHandler.setText(label_text)
    
    def addPointTag(self, pos):
        topLeft = QPointF(pos.x() - self.tag_rect_size / 2, pos.y() - self.tag_rect_size / 2)
        tagItem = TagPoint(QRectF(topLeft, QSizeF(self.tag_rect_size, self.tag_rect_size)), self)
        tagItem.setFlags(QGraphicsItem.ItemIsMovable|QGraphicsItem.ItemIsSelectable)
        tagItem.setOpacity(1)

        tagText = QGraphicsTextItem(str(self.getNewIndex()), parent = tagItem)   
        tagText.setFont(QFont('Arial', self.fontSize, QFont.Bold, False))
        tagText.setDefaultTextColor(QColor(Qt.green))
        tagText.document().setDocumentMargin(0)
        textPos = tagItem.mapFromScene(pos)
        tagText.setPos(textPos.x() - tagText.boundingRect().width(),
                        textPos.y() - tagText.boundingRect().height())
        imagePos = self.mapFromScene(pos).toPoint()
        self._TagDataTableModel.insertImagePoint(self.frame_Number, self.fileName, int(tagText.toPlainText()), pos.x(), pos.y())
        
        return tagItem
    
    def resortTagsText(self):
        for index, tagItem in enumerate(self.childItems()):
            tagItem.childItems()[0].setPlainText(str(index + 1))
            
    def getImageHeightWidth(self):
        imageSizeStr = "{0:d},{1:d}".format(self.pixmap().height(),self.pixmap().width())
        return imageSizeStr
            

class GraphicsSceneWithTag(QGraphicsScene):
    def __init__(self, parent = None):
        super(GraphicsSceneWithTag, self).__init__(parent = parent)
        self.currentFrame = 0
        self.imageLayer = []
        self._TagDataTableModel = None
        self.showImageItem = None
        self._tagEnabled = False
        self._posIndicatorHandler = None
    
    def reinitScene(self):
        self.clear()
        self.currentFrame = 0
        self.imageLayer[:] = []
    
    def setPosIndicatorHandler(self, pos_label):
        self._posIndicatorHandler = pos_label
    
    def enableTag(self):
        self._tagEnabled = True
        for a_image_item in self.imageLayer:
            a_image_item.enableTag()
        
    def disableTag(self):
        self._tagEnabled = False
        for a_image_item in self.imageLayer:
            a_image_item.disableTag()        

    def setTagDataTableModel(self, dataTableModel):
        self._TagDataTableModel = dataTableModel

    def showFrame(self, frame):
        childrenCount = len(self.imageLayer)
        if childrenCount > 0:
            if frame in range(0, childrenCount):
                for aItem in self.imageLayer:
                    aItem.setAcceptHoverEvents(False)
                    aItem.hide()

                self.showImageItem = self.imageLayer[frame]
                self.showImageItem.show()
                self.showImageItem.setAcceptHoverEvents(True)
                self.currentFrame = frame

    def loadImageList(self, imageFileDir, imageFileList):
        imageCount = 0
        imageFileList = sorted(imageFileList)
        image_size_strings = []
        for aFile in imageFileList:
            addedPixmapItem = ImageSceneItem(QPixmap(os.path.join(imageFileDir, aFile)), frameNumber = imageCount, fileName = aFile)
            image_size_strings.append(addedPixmapItem.getImageHeightWidth())
            self.addItem(addedPixmapItem)
            self.imageLayer.append(addedPixmapItem)            
            addedPixmapItem.setTagDataTableModel(self._TagDataTableModel)
            addedPixmapItem.setPosIndicatorHandler(self._posIndicatorHandler)
            addedPixmapItem.hide()
            imageCount += 1
            
        if len(image_size_strings) > 0:
            image_height_width_str = image_size_strings[0]
        else:
            image_height_width_str = ''
            
        return imageCount, image_height_width_str
    

    def contextMenuEvent(self, event):
        if self._tagEnabled:
            if len(self.selectedItems()) > 0:
                contextMenu = QMenu(self.views()[0])
                deleteAction = contextMenu.addAction('Delete')
                deleteAction.triggered.connect(self.removeSelectedTagItems)
                contextMenu.exec_(event.screenPos())
    
    def mouseReleaseEvent(self, event):
        if self._tagEnabled:
            if event.modifiers() == Qt.ShiftModifier and event.button() == Qt.RightButton:
                self.removeSingleTagItem(event.scenePos())
            elif event.button() == Qt.LeftButton:            
                    self.updateTagItemPos(event.scenePos())            
        QGraphicsScene.mouseReleaseEvent(self, event)

    def removeSingleTagItem(self, pos):
        currentItem = self.itemAt(pos, self.views()[0].viewportTransform())
        self.removeTagItem(currentItem)
    
    def removeSelectedTagItems(self):
        for item in self.selectedItems():
            self.removeTagItem(item)
    
    def removeTagItem(self, currentItem):
        if currentItem is not None and currentItem.__class__.__name__ == 'TagPoint':
            FrameNumber = currentItem.parentItem().frame_Number
            CorrespondingID = currentItem.childItems()[0].toPlainText()
            self._TagDataTableModel.removePointsInImage(FrameNumber, CorrespondingID)
            self.removeItem(currentItem)
            self.showImageItem.resortTagsText()
            
    def updateTagItemPos(self, pos):
        currentItem = self.itemAt(pos, self.views()[0].viewportTransform())
        if currentItem is not None and currentItem.__class__.__name__ == 'TagPoint':
            CorrespondingID = currentItem.childItems()[0].toPlainText()
            self._TagDataTableModel.updateTagPointPosition(currentItem.parentItem().frame_Number, CorrespondingID, pos.x(), pos.y())  
            

class SelectedPointsTableModel(QObject):
    def __init__(self, parent = None, databaseCon = None, work_dir=''):
        super(SelectedPointsTableModel, self).__init__(parent)
        self._DatabaseConn = databaseCon
        self._PointsInImages_SQLTableModel = QSqlTableModel(self, databaseCon)
        self._PointsInScene_SQLTableModel = QSqlTableModel(self, databaseCon)
        self._configOps = ConfigureOperators.ConfigureOps(work_dir)
        self._InImagePoints_TN = self._configOps.getTNTaggedImagePoints()
        self._InScenePoints_TN = self._configOps.getTNTaggedWorldPoints()

        self._PointsInImages_SQLTableModel.setTable(self._InImagePoints_TN)
        self._PointsInScene_SQLTableModel.setTable(self._InScenePoints_TN)

        self._PointsInImages_SQLTableModel.setEditStrategy(QSqlTableModel.OnManualSubmit)
        self._PointsInScene_SQLTableModel.setEditStrategy(QSqlTableModel.OnManualSubmit)     
        
        self._sessionID = ''
        self._operationID = 0

        self._DataViewQueryModel = QSqlQueryModel()

        self.toViewModel = QSortFilterProxyModel()
        self.toViewModel.setSourceModel(self._DataViewQueryModel)

        
        self.initializeTableModel()

        self.toViewModel.setDynamicSortFilter(True)
        self.refreshViewDataModel()
        
    def refreshDataQuery(self, session_id, operation_id):
        if len(session_id) > 0 and operation_id > 0:
            self._sessionID = session_id
            self._operationID = operation_id
            dataViewQueryCmd = ("SELECT {0}.ID as ID, {0}.SessionID as SessionID, {0}.OperationID as OperationID, {0}.FrameNumber as FrameNumber,"
                             " {0}.ImageFileName as ImageFileName, {0}.CorrespondingID as CorrespondingID,"
                             " {0}.ImagePoint_x as ImagePoint_x, {0}.ImagePoint_y as ImagePoint_y, {1}.ScenePoint_x as ScenePoint_x,"
                             " {1}.ScenePoint_y as ScenePoint_y, {1}.ScenePoint_z as ScenePoint_z"
                             " FROM {0} INNER JOIN {1} on {0}.CorrespondingID = {1}.CorrespondingID"
                             " WHERE {0}.SessionID='{2}' and {0}.OperationID={3};").format(self._InImagePoints_TN,
                                                                                           self._InScenePoints_TN,
                                                                                           self._sessionID,
                                                                                           self._operationID)
            self._DataViewQueryModel.setQuery(dataViewQueryCmd, self._DatabaseConn)
            self._DataViewQueryModel.submit()
            return True
        else:
            return False
        
    
    def initializeTableModel(self):
        self._DataViewQueryModel.setHeaderData(0, Qt.Horizontal, "ID")
        self._DataViewQueryModel.setHeaderData(1, Qt.Horizontal, "SessionID")
        self._DataViewQueryModel.setHeaderData(2, Qt.Horizontal, "OperationID")
        self._DataViewQueryModel.setHeaderData(3, Qt.Horizontal, "Frame_Number")
        self._DataViewQueryModel.setHeaderData(4, Qt.Horizontal, "Image_Name")
        self._DataViewQueryModel.setHeaderData(5, Qt.Horizontal, "Corresponding_ID")
        self._DataViewQueryModel.setHeaderData(6, Qt.Horizontal, "ImagePoint_x")
        self._DataViewQueryModel.setHeaderData(7, Qt.Horizontal, "ImagePoint_y")
        self._DataViewQueryModel.setHeaderData(8, Qt.Horizontal, "ScenePoint_x")
        self._DataViewQueryModel.setHeaderData(9, Qt.Horizontal, "ScenePoint_y")
        self._DataViewQueryModel.setHeaderData(10, Qt.Horizontal, "ScenePoint_z")

        
    def insertImagePoint(self, FrameNumber, Image_Name, CorrespondingID, x, y):
        new_filter = "SessionID='{0}' and OperationID={1} and CorrespondingID={2} and FrameNumber={3}".format(self._sessionID,
                                                                                                              self._operationID,
                                                                                                              CorrespondingID,
                                                                                                              FrameNumber)
        self._PointsInImages_SQLTableModel.setFilter(new_filter)
        self._PointsInImages_SQLTableModel.select()
        row_count = self._PointsInImages_SQLTableModel.rowCount()
        if  row_count== 0:
            newRecord = self._PointsInImages_SQLTableModel.record()
            newRecord.setValue("SessionID", self._sessionID)
            newRecord.setValue("OperationID", self._operationID)
            newRecord.setValue("FrameNumber", FrameNumber)
            newRecord.setValue("ImageFileName", Image_Name)
            newRecord.setValue("CorrespondingID", CorrespondingID)
            newRecord.setValue("ImagePoint_x", x)
            newRecord.setValue("ImagePoint_y", y)

            if not self._PointsInImages_SQLTableModel.insertRecord(row_count, newRecord):
                QMessageBox.critical(None,
                                     "Insert Data Error!",
                                     "Database Error: %s" % self._PointsInImages_SQLTableModel.lastError().text()
                                    )
                
                return               
            
        elif row_count > 0:
            matchedRecord = self._PointsInImages_SQLTableModel.record(row_count-1)
            matchedRecord.setValue("ImagePoint_x", x)
            matchedRecord.setValue("ImagePoint_y", y)
            self._PointsInImages_SQLTableModel.setRecord(row_count-1, matchedRecord)
        

        self._PointsInImages_SQLTableModel.submitAll()
        self._PointsInImages_SQLTableModel.setFilter("")
        self._PointsInImages_SQLTableModel.select()        
        self.refreshViewDataModel()
    
    def updateTagPointPosition(self, FrameNumber, CorrespondingID, x, y):
        
        new_filter = "SessionID='{0}' and OperationID={1} and CorrespondingID={2} and FrameNumber={3}".format(self._sessionID,
                                                                                                              self._operationID,
                                                                                                              CorrespondingID,
                                                                                                              FrameNumber)
        self._PointsInImages_SQLTableModel.setFilter(new_filter)
        self._PointsInImages_SQLTableModel.select()
        row_count = self._PointsInImages_SQLTableModel.rowCount()
        
        if row_count > 0:
            matchedRecord = self._PointsInImages_SQLTableModel.record(row_count - 1)
            matchedRecord.setValue("ImagePoint_x", x)
            matchedRecord.setValue("ImagePoint_y", y)
            self._PointsInImages_SQLTableModel.setRecord(row_count - 1, matchedRecord)
                
            self._PointsInImages_SQLTableModel.submitAll()
            self._PointsInImages_SQLTableModel.setFilter("")
            self._PointsInImages_SQLTableModel.select()             
            self.refreshViewDataModel()               

        
    def insertScenePoint(self, CorrespondingID, x, y, z):
        
        new_filter = "SessionID='{0}' and OperationID={1} and CorrespondingID={2}".format(self._sessionID,
                                                                                          self._operationID,
                                                                                          CorrespondingID)
        self._PointsInScene_SQLTableModel.setFilter(new_filter)
        self._PointsInScene_SQLTableModel.select()
        row_count = self._PointsInScene_SQLTableModel.rowCount()
        if row_count > 0:
            for rowIndex in matchedRowIndices:
                matchedRecord = self._PointsInScene_SQLTableModel.record(row_count - 1)
                matchedRecord.setValue("ScenePoint_x", x)
                matchedRecord.setValue("ScenePoint_y", y)
                matchedRecord.setValue("ScenePoint_z", z)
                self._PointsInScene_SQLTableModel.setRecord(row_count - 1, matchedRecord)
            
        else:
            newRecord = self._PointsInScene_SQLTableModel.record()
            newRecord.setValue("SessionID", self._sessionID)
            newRecord.setValue("OperationID", self._operationID)
            newRecord.setValue("CorrespondingID", CorrespondingID)
            newRecord.setValue("ScenePoint_x", x)
            newRecord.setValue("ScenePoint_y", y)
            newRecord.setValue("ScenePoint_z", z)

            if not self._PointsInScene_SQLTableModel.insertRecord(row_count, newRecord):
                QMessageBox.critical(None,
                                     "Insert Data Error!",
                                     "Database Error: %s" % self._PointsInScene_SQLTableModel.lastError().text()
                                    )
                return
            
        self._PointsInScene_SQLTableModel.submitAll()
        self._PointsInScene_SQLTableModel.setFilter("")
        self._PointsInScene_SQLTableModel.select()
        self.refreshViewDataModel()

    def refreshViewDataModel(self):
        thisQuery = self._DataViewQueryModel.query()
        thisQuery.exec_()
        self._DataViewQueryModel.setQuery(thisQuery)
        self.toViewModel.sort(0, Qt.AscendingOrder)        

    def removePointsInImage(self, FrameNumber, CorrespondingID):
        remove_filter = "SessionID='{0}' and OperationID={1} and CorrespondingID={2} and FrameNumber={3}".format(self._sessionID,
                                                                                                                 self._operationID,
                                                                                                                 CorrespondingID,
                                                                                                                 FrameNumber)
        self._PointsInImages_SQLTableModel.setFilter(remove_filter)
        self._PointsInImages_SQLTableModel.select()
        remove_rowCount = self._PointsInImages_SQLTableModel.rowCount()
        print "Debug removePointsInImage:"
        print "To remove row count: {0}".format(remove_rowCount)
        if remove_rowCount > 0: 
            ret = self._PointsInImages_SQLTableModel.removeRows(0, remove_rowCount)
            print "Remove op return"
            print ret
            
            ret = self._PointsInImages_SQLTableModel.submitAll()
            print "Submit all op return"
            print ret            
            self._PointsInImages_SQLTableModel.setFilter("")
            self.resortPointsInImage(FrameNumber)
            self.refreshViewDataModel()

    def removePointsInScene(self, CorrespondingID):
        remove_filter = "SessionID='{0}' and OperationID={1} and CorrespondingID={2}".format(self._sessionID,
                                                                                             self._operationID,
                                                                                             CorrespondingID)
        self._PointsInScene_SQLTableModel.setFilter(remove_filter)
        self._PointsInScene_SQLTableModel.select()
        remove_rowCount = self._PointsInScene_SQLTableModel.rowCount()
        if remove_rowCount > 0:
            self._PointsInScene_SQLTableModel.removeRows(0, remove_rowCount)

        self._PointsInScene_SQLTableModel.submitAll()
        self._PointsInScene_SQLTableModel.setFilter("")
        self.resortPointsInScene()
    
    def resortPointsInImage(self, FrameNumber):
        resort_filter = "SessionID='{0}' and OperationID={1} and FrameNumber={2}".format(self._sessionID, self._operationID, str(FrameNumber))
        self._PointsInImages_SQLTableModel.setFilter(resort_filter)
        self._PointsInImages_SQLTableModel.select()
        self._PointsInImages_SQLTableModel.sort(3, Qt.AscendingOrder)

        for i in range(self._PointsInImages_SQLTableModel.rowCount()):
            toUpdateRecord = self._PointsInImages_SQLTableModel.record(i)
            toUpdateRecord.setValue("CorrespondingID", i + 1)
            self._PointsInImages_SQLTableModel.setRecord(i, toUpdateRecord)
        self._PointsInImages_SQLTableModel.submitAll()
        
        self._PointsInImages_SQLTableModel.setFilter("")
        self._PointsInImages_SQLTableModel.select()
        self.refreshViewDataModel()

    def resortPointsInScene(self):
        self._PointsInScene_SQLTableModel.setFilter("SessionID='{0}' and OperationID={1}".format(self._sessionID, self._operationID))
        self._PointsInScene_SQLTableModel.select()
        self._PointsInScene_SQLTableModel.sort(2, Qt.AscendingOrder)

        for i in range(self._PointsInScene_SQLTableModel.rowCount()):
            toUpdateRecord = self._PointsInScene_SQLTableModel.record(i)
            toUpdateRecord.setValue("CorrespondingID", i + 1)
            self._PointsInScene_SQLTableModel.setRecord(i, toUpdateRecord)
        self._PointsInScene_SQLTableModel.submitAll()
        
        self._PointsInScene_SQLTableModel.setFilter("")
        self._PointsInScene_SQLTableModel.select()
        self.refreshViewDataModel()
        
    def resetCurrentTagOperation(self):
        delete_image_points_cmd = "DELETE FROM {0} WHERE SessionID='{1}' and OperationID={2}".format(self._InImagePoints_TN,self._sessionID, self._operationID)
        delete_scene_points_cmd = "DELETE FROM {0} WHERE SessionID='{1}' and OperationID={2}".format(self._InScenePoints_TN,self._sessionID, self._operationID)
        delete_query = QSqlQuery(self._DatabaseConn)
        delete_query.exec_(delete_image_points_cmd)
        delete_query.exec_(delete_scene_points_cmd)
        self._PointsInImages_SQLTableModel.select()
        self._PointsInScene_SQLTableModel.select()
        self.refreshViewDataModel()

    def submitDataChanges(self):
        self._PointsInImages_SQLTableModel.submitAll()
        self._PointsInScene_SQLTableModel.submitAll()
        
        
    def get_user_image_ponits(self):
        user_image_points_list = []
        user_image_data_query = QSqlQuery(self._DatabaseConn)
        if user_image_data_query.exec_("SELECT * FROM {0}".format(self._InImagePoints_TN)):
            frame_index = user_image_data_query.record().indexOf("FrameNumber")
            corres_id_index = user_image_data_query.record().indexOf("CorrespondingID")
            x_index = user_image_data_query.record().indexOf("ImagePoint_x")
            y_index = user_image_data_query.record().indexOf("ImagePoint_y")
                
            while user_image_data_query.next():
                frame = user_image_data_query.value(frame_index)
                c_id = user_image_data_query.value(corres_id_index)
                x = user_image_data_query.value(x_index)
                y = user_image_data_query.value(y_index)
                user_image_points_list.append(
                    {
                        "frame": frame,
                        "c_id": c_id,
                        "x": x,
                        "y": y
                     }
                )
            
        return user_image_points_list
    
    def get_user_world_points(self):
        user_world_points_list = []
        user_world_data_query = QSqlQuery(self._DatabaseConn)
        if user_world_data_query.exec_("SELECT * FROM {0}".format(self._InScenePoints_TN)):
            corres_id_index = user_world_data_query.record().indexOf("CorrespondingID")
            x_index = user_world_data_query.record().indexOf("ScenePoint_x")
            y_index = user_world_data_query.record().indexOf("ScenePoint_y")
            z_index = user_world_data_query.record().indexOf("ScenePoint_z")
            
            while user_world_data_query.next():              
                c_id = user_world_data_query.value(corres_id_index)
                x = user_world_data_query.value(x_index)
                y = user_world_data_query.value(y_index)
                z = user_world_data_query.value(z_index)
                user_world_points_list.append(
                    {
                        "c_id": c_id,
                        "x": x,
                        "y": y,
                        "z": z
                     }
                )        
        
        return user_world_points_list
    
    def sendTaggedPoints(self):        
        get_frames_cmd = ("SELECT DISTINCT FrameNumber FROM {0}"
                          " WHERE SessionID='{1}' AND OperationID={2}"
                          " ORDER BY FrameNumber ASC;").format(self._InImagePoints_TN, self._sessionID, self._operationID)
        frame_query = QSqlQuery(self._DatabaseConn)
        frame_query.exec_(get_frames_cmd)
        frames = []
        while frame_query.next():
            frames.append(frame_query.value(0))
        frame_query.finish()
        features_dict = {}
        operation_dict = {}
        
        world_points_filter = ("SessionID='{0}' and OperationID={1}").format(self._sessionID, self._operationID)
        self._PointsInScene_SQLTableModel.setFilter(world_points_filter)
        self._PointsInScene_SQLTableModel.setSort(3, Qt.AscendingOrder)
        self._PointsInScene_SQLTableModel.select()
        
        wp_count = self._PointsInScene_SQLTableModel.rowCount()

        for i in range(wp_count):
            curr_record = self._PointsInScene_SQLTableModel.record(i)
            corresponding_id = curr_record.value('CorrespondingID')
            
            image_points_filter = ("SessionID='{0}' and OperationID={1} and CorrespondingID={2}").format(self._sessionID, self._operationID, corresponding_id)
            
            self._PointsInImages_SQLTableModel.setFilter(image_points_filter)
            self._PointsInImages_SQLTableModel.setSort(4, Qt.AscendingOrder)
            self._PointsInImages_SQLTableModel.select()
            ip_count = self._PointsInImages_SQLTableModel.rowCount()        
            image_point_dict = {}            
            for j in range(ip_count):
                curr_image_record = self._PointsInImages_SQLTableModel.record(j)
                image_point_dict.setdefault(str(curr_image_record.value('FrameNumber')),
                                            {'file_name':  curr_image_record.value('ImageFileName'),
                                             'image_point': [curr_image_record.value('ImagePoint_x'),
                                                             curr_image_record.value('ImagePoint_y')]})
                
            image_point_dict.setdefault('world_point', [curr_record.value('ScenePoint_x'),
                                                        curr_record.value('ScenePoint_y'),
                                                        curr_record.value('ScenePoint_z')])
            
            features_dict.setdefault(str(curr_record.value('CorrespondingID')), image_point_dict)
            
        operation_dict.setdefault('session_id', self._sessionID)
        operation_dict.setdefault('operation_id', self._operationID)
        operation_dict.setdefault('features', features_dict)
        operation_dict.setdefault('frames', frames)
        
        session_dir = os.path.join(self._configOps.getServerCachePath(), self._sessionID)
        if not os.path.isdir(session_dir):        
            os.mkdir(session_dir)
        user_tag_file = os.path.join(session_dir, self._configOps.getUserTaggedJsonFile())
        
        with open(user_tag_file, 'w') as wfp:
            dump(operation_dict, wfp)
            
        return True
        

class DatabaseManager(QObject):
    def __init__(self, parent=None, work_dir = ''):
        super(DatabaseManager, self).__init__(parent=parent)
        self.workdir = work_dir
        self._configOps = ConfigureOperators.ConfigureOps(self.workdir)
        
        self.isDatabaseConnected = False
        self.isTableInitialized = [False, False, False, False]
        self.defaultConnection = None
        self.connectionName = 'defaultConnection'

        self._InImagePoints_TN = self._configOps.getTNTaggedImagePoints()
        self._InScenePoints_TN = self._configOps.getTNTaggedWorldPoints()
        self._sessionManagement_TN = self._configOps.getTNSessionManagement()
        self._operations_TN = self._configOps.getTNOperations()

    def getDefaultConnection(self):
        databaseName = os.path.join(self.workdir, self._configOps.getMayaInterfaceDatabaseName())
        db_dir_name = os.path.dirname(databaseName)
        
        if not os.path.isdir(db_dir_name):
            os.mkdir(db_dir_name)
        
        self.defaultConnection = QSqlDatabase.addDatabase('QSQLITE', self.connectionName)
        self.defaultConnection.setDatabaseName(databaseName)
        if not self.defaultConnection.open():
            QMessageBox.critical(
                None,
                "Database Creating Error",
                "Database Error: %s" % self.defaultConnection.lastError().databaseText()
            )
            return None

        self.isDatabaseConnected = True

        return self.defaultConnection

    def initialTables(self):
        if self.defaultConnection.isOpen():
            createTableQuery = QSqlQuery(self.defaultConnection)
            
            self.isTableInitialized[0] = createTableQuery.exec_(
                                            """
                                            CREATE TABLE IF NOT EXISTS "{0}" (
                                                "ID" INTEGER NOT NULL UNIQUE,
                                                "SessionID" TEXT,
                                                PRIMARY KEY("ID" AUTOINCREMENT)
                                            );""".format(self._sessionManagement_TN)
                                        )
            
            self.isTableInitialized[1] = createTableQuery.exec_(
                                            """
                                            CREATE TABLE IF NOT EXISTS "{0}" (
                                                "ID" INTEGER NOT NULL UNIQUE,
                                                "SessionID" TEXT,
                                                "OperationName" TEXT,
                                                "SFMServerStatus" TEXT,
                                                PRIMARY KEY("ID" AUTOINCREMENT)
                                            );""".format(self._operations_TN)
                                        )            
            
            self.isTableInitialized[2] = createTableQuery.exec_(
                                            """
                                            CREATE TABLE IF NOT EXISTS "{0}" (
                                                "ID" INTEGER NOT NULL UNIQUE,
                                                "SessionID" TEXT,
                                                "OperationID" INTEGER,
                                                "CorrespondingID" INTEGER,
                                                "FrameNumber" INTEGER,
                                                "ImageFileName" TEXT,
                                                "ImagePoint_x" INTEGER,
                                                "ImagePoint_y" INTEGER,
                                                PRIMARY KEY("ID" AUTOINCREMENT)
                                            );""".format(self._InImagePoints_TN)
                                        )
            self.isTableInitialized[3] = createTableQuery.exec_(
                                            """
                                            CREATE TABLE IF NOT EXISTS "{0}" (
                                                "ID" INTEGER NOT NULL UNIQUE,
                                                "SessionID" TEXT,
                                                "OperationID" INTEGER,
                                                "CorrespondingID" INTEGER,
                                                "ScenePoint_x" REAL,
                                                "ScenePoint_y" REAL,
                                                "ScenePoint_z" REAL,
                                                PRIMARY KEY("ID" AUTOINCREMENT)
                                            );""".format(self._InScenePoints_TN)
                                        )
            createTableQuery.finish()
            
    def get_user_image_ponits(self):
        user_image_points_list = []
        user_image_data_query = QSqlQuery(self.defaultConnection)
        if user_image_data_query.exec_("SELECT * FROM {0}".format(self._InImagePoints_TN)):
            frame_index = user_image_data_query.record().indexOf("FrameNumber")
            corres_id_index = user_image_data_query.record().indexOf("CorrespondingID")
            x_index = user_image_data_query.record().indexOf("ImagePoint_x")
            y_index = user_image_data_query.record().indexOf("ImagePoint_y")
                
            while user_image_data_query.next():
                frame = user_image_data_query.value(frame_index)
                c_id = user_image_data_query.value(corres_id_index)
                x = user_image_data_query.value(x_index)
                y = user_image_data_query.value(y_index)
                user_image_points_list.append(
                    {
                        "frame": frame,
                        "c_id": c_id,
                        "x": x,
                        "y": y
                     }
                )
            
        return user_image_points_list
    
    def get_user_world_points(self):
        user_world_points_list = []
        user_world_data_query = QSqlQuery(self.defaultConnection)
        if user_world_data_query.exec_("SELECT * FROM {0}".format(self._InScenePoints_TN)):
            corres_id_index = user_world_data_query.record().indexOf("CorrespondingID")
            x_index = user_world_data_query.record().indexOf("ScenePoint_x")
            y_index = user_world_data_query.record().indexOf("ScenePoint_y")
            z_index = user_world_data_query.record().indexOf("ScenePoint_z")
            
            while user_world_data_query.next():              
                c_id = user_world_data_query.value(corres_id_index)
                x = user_world_data_query.value(x_index)
                y = user_world_data_query.value(y_index)
                z = user_world_data_query.value(z_index)
                user_world_points_list.append(
                    {
                        "c_id": c_id,
                        "x": x,
                        "y": y,
                        "z": z
                     }
                )        
        
        return user_world_points_list
    
    def getSessions(self):
        existing_session_list = []
        session_list_query = QSqlQuery(self.defaultConnection)
        if session_list_query.exec_("SELECT SessionID from {0}".format(self._sessionManagement_TN)):
            while session_list_query.next():
                existing_session_list.append(session_list_query.value(0))

        return existing_session_list
    
    
    def getOperations(self, session_id):
        operation_list = []
        operation_list_query = QSqlQuery(self.defaultConnection)
        sql_cmd = "SELECT ID, OperationName FROM Operations WHERE SessionID='{0}' ORDER BY ID DESC".format(session_id)
        if operation_list_query.exec_(sql_cmd):
            while operation_list_query.next():
                operation_list.append(": ".join([str(operation_list_query.value(0)), operation_list_query.value(1)]))
        return operation_list
    
    def newSession(self, session_id):
        insert_session_query = QSqlQuery(self.defaultConnection)
        sql_cmd = "INSERT INTO {0} (SessionID) VALUES ('{1}')".format(self._configOps.getTNSessionManagement(), session_id)
        if insert_session_query.exec_(sql_cmd):
            print "Database log {0}".format(session_id)
        else:
            print "Something wrong with Database!"
            print sql_cmd
            
    def newTagOperation(self, session_id):
        operation_query = QSqlQuery(self.defaultConnection)
        sql_cmd = "INSERT INTO {0} (SessionID, OperationName) VALUES ('{1}', '{2}');".format(self._configOps.getTNOperations(), session_id, OPERATION.TAG_POINTS)
        if operation_query.exec_(sql_cmd):
            operation_query.finish()
            operation_query.exec_("SELECT ID from {0} WHERE SessionID='{1}' ORDER BY ID DESC LIMIT 1;".format(self._configOps.getTNOperations(), session_id))
            operation_query.first()
            last_op_id = operation_query.value(0)
            print "Database log Operation {0}".format(last_op_id)
            return last_op_id            
        else:
            print "Something wrong with Database!"
            print sql_cmd
        
    
    def clearData(self, session_id):
        if self.defaultConnection is None:
            return
        if self.defaultConnection.isOpen():
            toClearDatabaseQuery = QSqlQuery(self.defaultConnection)
            toClearDatabaseQuery.exec_(
                "DELETE from {0} WHERE SessionID='{1}';".format(self._InImagePoints_TN, session_id)
            )

            toClearDatabaseQuery.exec_(
                "DELETE from {0} WHERE SessionID='{1}';".format(self._InScenePoints_TN, session_id)
            )
            
            toClearDatabaseQuery.exec_(
                "DELETE from {0} WHERE SessionID='{1}';".format(self._operations_TN, session_id)
            )
            
            toClearDatabaseQuery.exec_(
                "DELETE from {0} WHERE SessionID='{1}';".format(self._sessionManagement_TN, session_id)
            )               

            toClearDatabaseQuery.finish()
        
    def closeDatabase(self):
        if self.defaultConnection.isOpen():
            self.defaultConnection.close()
            QSqlDatabase.removeDatabase(self.connectionName)
        


