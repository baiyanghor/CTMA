import argparse
import sys

work_dir = "E:/VHQ/camera-track-study/track-with-model/CameraTrackModelAlignment_py310_operationally/cam_track_service"
if work_dir not in sys.path:
    sys.path.append(work_dir)

from SFMOperations.SFMPipelineInterface import SFMPipelineInterface

sfm_service_parser = argparse.ArgumentParser(description='SFM Service Interface')
sfm_service_parser.add_argument('--new_session', action='store', nargs=2, help='Create new session from user.')
sfm_service_parser.add_argument('--get_initial_pair', action='store', help='Initial image pair selected for SFM.')
sfm_service_parser.add_argument('--set_sfm_global_info', action='store', nargs=5,
                                help='Location where image sequence stored.')
sfm_service_parser.add_argument('--get_new_session_id', action='store_true', help='Get new session ID.')
sfm_service_parser.add_argument('--user_correspond_ready', action='store',
                                help='Notify SFM server user has selected correspond points.')
sfm_service_parser.add_argument('--set_image_point', action='store', nargs=5, help='Set image points.')
sfm_service_parser.add_argument('--set_world_point', action='store', nargs=5, help='Set world points.')
sfm_service_parser.add_argument('--set_user_tagged_done', action='store', nargs=2, help='Set user tagged points done.')
sfm_service_parser.add_argument('--initial_pnp_extrinsic', action='store',
                                help='Kick out initial PNP extrinsic calculation.')

request_args = sfm_service_parser.parse_args()

if request_args.new_session:
    new_session_data = request_args.new_session
    sfm_interface = SFMPipelineInterface(work_dir)
    session_id = sfm_interface.new_session(*new_session_data)
    sys.stdout.write(session_id)

if request_args.set_sfm_global_info:
    sfm_global_info_data = request_args.set_sfm_global_info
    sfm_interface = SFMPipelineInterface(work_dir)
    if sfm_interface.set_sfm_global_info(*sfm_global_info_data):
        sys.stdout.write('OK')
    else:
        sys.stdout.write('Failure')

if request_args.get_initial_pair:
    sfm_interface = SFMPipelineInterface(work_dir)
    session_id = request_args.get_initial_pair
    sfm_initial_pair = sfm_interface.get_initial_sfm_pair(session_id)
    if isinstance(sfm_initial_pair, tuple):
        sys.stdout.write(','.join([str(sfm_initial_pair[0]), str(sfm_initial_pair[1])]))
    else:
        sys.stdout.write('Failure')

if request_args.set_image_point:
    image_point_data = request_args.set_image_point
    sfm_interface = SFMPipelineInterface(work_dir)
    if sfm_interface.set_user_selected_image_point(*image_point_data):
        sys.stdout.write('OK')
    else:
        sys.stdout.write('Failure')


if request_args.set_world_point:
    world_point_data = request_args.set_world_point
    sfm_interface = SFMPipelineInterface(work_dir)
    if sfm_interface.set_user_selected_world_point(*world_point_data):
        sys.stdout.write('OK')
    else:
        sys.stdout.write('Failure')

if request_args.set_user_tagged_done:
    session_id_op_id = request_args.set_user_tagged_done
    sfm_interface = SFMPipelineInterface(work_dir)
    if sfm_interface.set_user_tagged_done(*session_id_op_id):
        sys.stdout.write('OK')
    else:
        sys.stdout.write('Failure')

if request_args.initial_pnp_extrinsic:
    session_id = request_args.initial_pnp_extrinsic
    sfm_interface = SFMPipelineInterface(work_dir)

    if sfm_interface.initial_pnp_extrinsic(session_id):
        sys.stdout.write('OK')
    else:
        sys.stdout.write('Failure')


