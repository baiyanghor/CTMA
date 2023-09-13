import os
import json

class ConfigureOps():
    
    def __init__(self, workingDir = ''):
        self.configureFile = os.path.join(os.path.abspath(workingDir), "Configure/configure.json")
        self.configureData = None
        if os.path.exists(self.configureFile):
            with open(self.configureFile, 'r') as fileHandler:
                self.configureData = json.load(fileHandler)
        else:
            print("Debug: Configure file does not exists!")
            
        
        
    def getValidImageType(self):
        if self.configureData is not None:
            if 'ValidFileType' in self.configureData:
                return self.configureData['ValidFileType']
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None
        
        
    def getTempDataPath(self):
        if self.configureData is not None:
            if 'LocalTempDataPath' in self.configureData:
                return self.configureData['LocalTempDataPath']
            else:
                print("Debug: Data required not exists!")
                return None               
            
        else:
            print("Debug: Read json failure!")
            return None
    
    def getMayaInterfaceDatabaseName(self):
        if self.configureData is not None:
            if 'MayaOPDatabase' in self.configureData and 'DatabasePath' in self.configureData:
                return  os.path.join(self.configureData['DatabasePath'], self.configureData['MayaOPDatabase'])
            else:
                print("Debug: Data required not exists!")
                return None
        else:
            print("Debug: Read json failure!")
            return None

    def get_shadow_mask_path(self):
        if self.configureData is not None:
            if 'ShadowMaskPath' in self.configureData:
                return self.configureData['ShadowMaskPath']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None
        
    def getServerCachePath(self):
        if self.configureData is not None:
            if 'ServerTempPath' in self.configureData:
                return self.configureData['ServerTempPath']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None
        
        
    def getSFMServiceDir(self):
        if self.configureData is not None:
            if 'SFMServiceDir' in self.configureData:
                return self.configureData['SFMServiceDir']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None
        
    
    def getSFMCameraTranslationFileName(self):
        if self.configureData is not None:
            if 'CameraTranslationJsonFile' in self.configureData:
                return self.configureData['CameraTranslationJsonFile']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None
        
    def getSFMworldPointsFileName(self):
        if self.configureData is not None:
            if 'WorldPointsJsonFile' in self.configureData:
                return self.configureData['WorldPointsJsonFile']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None
    
    def getSessionFileName(self):
        if self.configureData is not None:
            if 'SessionStatusJsonFile' in self.configureData:
                return self.configureData['SessionStatusJsonFile']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None
        
    def getUserTaggedJsonFile(self):
        if self.configureData is not None:
            if 'UserTaggedJsonFile' in self.configureData:
                return self.configureData['UserTaggedJsonFile']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None     
        
    def getTNTaggedImagePoints(self):
        if self.configureData is not None:
            if 'TNTaggedImagePoints' in self.configureData:
                return self.configureData['TNTaggedImagePoints']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None
        
    def getTNTaggedWorldPoints(self):
        if self.configureData is not None:
            if 'TNTaggedWorldPoints' in self.configureData:
                return self.configureData['TNTaggedWorldPoints']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None
        
    
    def getTNSessionManagement(self):
        if self.configureData is not None:
            if 'TNSessionManagement' in self.configureData:
                return self.configureData['TNSessionManagement']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None        
    
    def getTNOperations(self):
        if self.configureData is not None:
            if 'TNOperations' in self.configureData:
                return self.configureData['TNOperations']
            else:
                print("Debug: Data required not exists!")
                return None

        else:
            print("Debug: Read json failure!")
            return None     
    
    
    