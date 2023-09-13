import argparse
import os
import sys
start_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_Py38_Algorithm_Update"
sys.path.append(start_dir)

from SFMOperations.SFMDatabaseInterface import SFMDatabaseInterface
from SFMOperations.SFMPipeline import SFMPipeLine

local_parser = argparse.ArgumentParser(description='Request SFM Data')
local_parser.add_argument('--get_initial_pair', action='store', help='Initial image pair selected for SFM.')
local_parser.add_argument('--set_images_path', action='store', nargs=2, help='Location where image sequence stored.')
local_parser.add_argument('--new_session', action='store', nargs=4, help='Create new session from user.')
local_parser.add_argument('--get_new_session_id', action='store_true', help='Get new session ID.')
local_parser.add_argument('--user_correspond_ready', action='store',
                          help='Notify SFM server user has selected correspond points.')
args = local_parser.parse_args()

work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sfm_pipeline = SFMPipeLine()

sfm_db_ops = SFMDatabaseInterface(work_dir)
sfm_db_ops.initial_sfm_database()


if args.get_initial_pair:
    session_id = str(args.get_initial_pair)
    sfm_pipeline.set_session_id(session_id)
    image_path = sfm_db_ops.get_images_path(session_id)
    sfm_pipeline.set_images_dir(image_path)
    sfm_pipeline.construct_sfm_image_list()
    initial_pair = sfm_pipeline.get_initial_sfm_pair()
    sfm_db_ops.set_image_pair(session_id, initial_pair)
    pair = sfm_db_ops.get_sfm_initial_pair(session_id)
    output_str = ','.join([str(value) for value in pair])
    sys.stdout.write(output_str)

if args.set_images_path:
    image_path_data = args.set_images_path
    sfm_db_ops.set_images_path(*image_path_data)
    sys.stdout.write('OK')

if args.new_session:
    new_session_data = args.new_session
    session_id = sfm_db_ops.new_session(*new_session_data)
    sys.stdout.write(session_id)

if args.get_new_session_id:
    new_id = sfm_db_ops.get_new_session_id()
    if new_id:
        sys.stdout.write(new_id)
    else:
        sys.stdout.write('Failure')

if args.user_correspond_ready:
    session_id = args.user_correspond_ready
    if sfm_db_ops.store_user_selected_ponits(session_id):
        sys.stdout.write('OK')
    else:
        sys.stdout.write('Failure!')



