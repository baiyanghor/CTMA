import os
import numpy as np

class TextFileOps(object):
    
    def saveArrayToTextFile(self, dataArray, fileName = ''):        
        if type(dataArray) == np.ndarray:
            if os.path.isfile(fileName):
                os.remove(fileName)

            np.savetxt(fileName, dataArray, fmt='%d')
            return True
        
        else:
            print("Debug: Support numpy.ndarray only")
            return False  
    
    def readArrayFromTextFile(self, fileName = ''):
        if os.path.isfile(fileName):
            arrayData = np.loadtxt(fileName)
            return arrayData
        
        else:
            print("Debug: Required file does not exists")
            return None
        
        
    def saveArrayToBFile(self, dataArray, fileName = ''):        
        if type(dataArray) == np.ndarray:
            if os.path.isfile(fileName):
                os.remove(fileName)
                
            np.save(fileName, dataArray)        
            return True
        
        else:
            print("Debug: Support numpy.ndarray only")
            return False
        
    def readArrayFromBFile(self, fileName = ''):
        if os.path.isfile(fileName):
            arrayData = np.load(fileName)
            return arrayData
        
        else:
            print("Debug: Required file does not exists")
            return None
        
    def saveFlattenData(self, dataArray, fileName = ''):
        if type(dataArray) == np.ndarray:
            if os.path.isfile(fileName):
                os.remove(fileName)
        else:
            print("Debug: Data type Error")
            return
        
        flattenArray = dataArray.flatten('C')
        np.savetxt(fileName, flattenArray)
        
        shapeFileName =os.path.join(os.path.dirname(fileName), os.path.basename(fileName).split('.')[0] + "_shape.txt") 
        
        if os.path.isfile(shapeFileName):
            os.remove(shapeFileName)
        
        with open(shapeFileName, "w") as wHandler:
            line = [str(dim) for dim in dataArray.shape]
            wHandler.write(','.join(line))
        
    
    
    
    
        
        