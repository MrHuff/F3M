from run_obj import *
import argparse
from generate_parameters import *

if __name__ == '__main__':
    fold = 'uniform_X_0_1'
    jobs = os.listdir(fold)
    for job in jobs:
        args = load_obj(job,folder=f'{fold}/')
        print(args)
        run_and_record_simulated(args_in=args)




