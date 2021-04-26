from run_obj import *
import argparse
from generate_parameters import *
def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_path', type=str, nargs='?', default='', help='which dataset to run')
    parser.add_argument('--idx', type=int, nargs='?', default=-1, help='which dataset to run')
    return parser


if __name__ == '__main__':
    input_args = vars(job_parser().parse_args())
    fold = input_args['job_path']
    idx = input_args['idx']
    jobs = os.listdir(fold)
    args = load_obj(jobs[idx],folder=f'{fold}/')
    run_and_record_simulated(args_in=args_in)




