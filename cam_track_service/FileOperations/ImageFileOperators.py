import os
from SFMConfigure import SFMConfigureOperators


class ImageFileOperators(object):
    def __init__(self, in_work_dir: str):
        self.workdir = in_work_dir
        cops = SFMConfigureOperators.SFMConfigureOps(self.workdir)
        self.validFileType = cops.get_valid_image_type()
    
    def getImageRawFileList(self, imageFileDir = ''):
        if os.path.isdir(imageFileDir):
            rawFileList = os.listdir(imageFileDir)
            rawFileList = sorted(rawFileList, reverse=False)
            imageNameList = [(lambda _:  os.path.join(imageFileDir, imageName))(imageName)
                             for imageName in filter(lambda x: x.split('.')[-1] in self.validFileType, rawFileList)]
            return imageNameList
        else:
            return None

