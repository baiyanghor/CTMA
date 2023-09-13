import subprocess

class SFMCalculatorInterface(object):
    
    def __init__(self, sfm_work_dir=''):
        self.sfm_work_dir = sfm_work_dir
        self.py3='"C:/Program Files/Python310/python.exe"'
        self.pyscript = self.sfm_work_dir + '/SFMOperations/SFMService.py'
    
    def get_initial_sfm_pair(self, session_id):        
        script_option = '--get_initial_pair'

        ret = subprocess.check_output(' '.join([self.py3, self.pyscript, script_option, session_id]), universal_newlines=True).strip()
        last_row = ret.split('\n')[-1]
        return last_row
            
    def set_global_sfm_info(self, session_id, focal_length, image_count, image_size, image_path):
        script_option = '--set_sfm_global_info'
        # try:
        ret = subprocess.check_output(' '.join([self.py3, self.pyscript, script_option, session_id, str(focal_length), str(image_count), image_size, image_path]),
                                      universal_newlines=True).strip()
        if ret == 'OK':
            return True
        else:
            return False
        
    def get_new_session_id(self):
        script_option = '--get_new_session_id'
        ret = subprocess.check_output(' '.join([self.py3, self.pyscript, script_option]), universal_newlines=True).strip()
        print("get_new_session_id")
        print(ret)        

        
    def new_session(self, user_name, user_location):
        script_option = '--new_session'
        # try:
        ret = subprocess.check_output(' '.join([self.py3, self.pyscript, script_option, user_name, \
            user_location]), universal_newlines=True).strip()
        return ret
        
    def set_user_world_point(self, session_id, c_id, x, y, z):
        script_option = '--set_world_point'
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        ret = subprocess.check_output(' '.join([self.py3, self.pyscript, script_option, session_id, \
                                                str(c_id), str(x), str(y), str(z)]),
                                                universal_newlines=True,
                                                startupinfo = si).strip()
        return ret
    
    def set_user_image_point(self, session_id, frame, c_id, x, y):
        script_option = '--set_image_point'
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE

        ret = subprocess.check_output(' '.join([self.py3, self.pyscript, script_option, session_id, \
                                                str(frame), str(c_id), str(x), str(y)]),
                                                universal_newlines=True,
                                                startupinfo = si).strip()
        return ret
    
    def send_user_defined_data(self, session_id, user_image_points,  user_world_points):
        if len(user_image_points) > 0:
            for a_user_image_point in user_image_points:
                ret = self.set_user_image_point(session_id,a_user_image_point['frame'],a_user_image_point['c_id'],a_user_image_point['x'],a_user_image_point['y'])
                if not ret=='OK':
                    print "Write user tagged image points failure!"
                    return False
                
        if len(user_world_points) > 0:
            for a_user_world_point in user_world_points:
                ret = self.set_user_world_point(session_id, a_user_world_point['c_id'], a_user_world_point['x'], a_user_world_point['y'], a_user_world_point['z'])
                if not ret=='OK':
                    print "Write user tagged world points failure!"
                    return False
        
        return True
    
    def set_user_tagged_done(self, session_id, op_id):
        """
        op_id = 0 for new user tagged operation
        """
        script_option = '--set_user_tagged_done'
        ret = subprocess.check_output(' '.join([self.py3, self.pyscript, script_option, session_id, op_id]), universal_newlines=True).strip()
        if ret == 'OK':
            return True
        else:
            return False
    
    def kickout_sfm_calculate(self, session_id, with_trajectory=0):
        pass
    
    def user_selected_ponits_ready(self, session_id):
        pass
        
        