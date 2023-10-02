# import sys
# import argparse
# from jai_benchmark import *
#
#
# if __name__ == '__main__':
#     print(f'argv: {sys.argv}')
#     # the cwd must be the root of the respository
#     if os.path.split(os.getcwd())[-1] == 'scripts':
#         os.chdir('../')
#     #
#
#     parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
#     parser.add_argument('settings_file', type=str, default=None)
#     parser.add_argument('--tensor_bits', type=utils.str_to_int)
#     parser.add_argument('--configs_path', type=str)
#     parser.add_argument('--models_path', type=str)
#     parser.add_argument('--task_selection', type=str, nargs='*')
#     parser.add_argument('--model_selection', type=str, nargs='*')
#     parser.add_argument('--model_exclusion', type=str, nargs='*')
#     parser.add_argument('--session_type_dict', type=str, nargs='*')
#     parser.add_argument('--num_frames', type=int)
#     parser.add_argument('--calibration_frames', type=int)
#     parser.add_argument('--calibration_iterations', type=int)
#     parser.add_argument('--run_import', type=utils.str_to_bool)
#     parser.add_argument('--run_inference', type=utils.str_to_bool)
#     parser.add_argument('--parallel_devices', type=int, nargs='*')
#     parser.add_argument('--modelartifacts_path', type=str)
#     cmds = parser.parse_args()
#     print(cmds)
#
#     kwargs = vars(cmds)
#     print(kwargs)
#     # print(**kwargs)
#     print(cmds.settings_file)

