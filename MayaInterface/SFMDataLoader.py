from json import load
import os
import maya.cmds as MCMD
import maya.api.OpenMaya as OM2
from Configure import ConfigureOperators
reload(ConfigureOperators)

class SFMDataLoader(object):
    def __init__(self, session_id='', work_dir='', camera_name='', locator_prefix= '', image_id_offset=0):
        self._session_id = session_id
        self._conf_ops = ConfigureOperators.ConfigureOps(work_dir)
        self.__camera_name = camera_name
        self.__locator_prefix = locator_prefix
        self._image_id_offset = image_id_offset
        
    def createSFMCameraInScene(self):
        if MCMD.objExists(self.__camera_name):
            MCMD.Delete(self.__camera_name)
        names = MCMD.camera()
        MCMD.rename(names[0], self.__camera_name)

    
    def keyCamera(self, frame, rotate, translate):
        MCMD.setKeyframe(self.__camera_name, t=frame, at='rotateX', v=rotate[0])
        MCMD.setKeyframe(self.__camera_name, t=frame, at='rotateY', v=rotate[1])
        MCMD.setKeyframe(self.__camera_name, t=frame, at='rotateZ', v=rotate[2])
        
        MCMD.setKeyframe(self.__camera_name, t=frame, at='translateX', v=translate[0])
        MCMD.setKeyframe(self.__camera_name, t=frame, at='translateY', v=translate[1])
        MCMD.setKeyframe(self.__camera_name, t=frame, at='translateZ', v=translate[2])
        
    
    def drawSFMCameras(self):
        sfm_server_cache_path = os.path.join(self._conf_ops.getServerCachePath(), self._session_id)
        translation_file_name = os.path.join(sfm_server_cache_path, self._conf_ops.getSFMCameraTranslationFileName())
        if os.path.isfile(translation_file_name):
            with open(translation_file_name, 'r') as filehandler:
                translation_list = load(filehandler)
                self.createSFMCameraInScene()
                if len(translation_list) > 0:
                    for atranslation in translation_list:
                        frame_num = atranslation.keys()[0]                    
                        sfm_rotate = atranslation.values()[0]['rotate']
                        sfm_translate = atranslation.values()[0]['translate']
                        self.keyCamera(int(frame_num) + self._image_id_offset, sfm_rotate, sfm_translate)
        else:
            print "Server data not ready yet!"
                    
    def drawWorldPoints(self):
        sfm_server_cache_path = os.path.join(self._conf_ops.getServerCachePath(), self._session_id)
        world_points_file_name = os.path.join(sfm_server_cache_path, self._conf_ops.getSFMworldPointsFileName())
        with open(world_points_file_name, 'r') as filehandler:
            world_points_data = load(filehandler)
            world_points_list = world_points_data['world_points']
            groupname = self.__locator_prefix + 'locators'
            MCMD.select(cl=True)
            MCMD.group(em=True, n=groupname)
            for i, pos in enumerate(world_points_list):
                locator_name = self.__locator_prefix + str(i + 1)
                MCMD.spaceLocator(n=locator_name, p=pos)
                MCMD.parent(locator_name, groupname)


