import torch
from conjugate_gradient.gp_experiment_object import experiment_object_gp
import argparse
import pickle
import os
# torch.multiprocessing.set_start_method("spawn")
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--job_folder', type=str, default='', help='cdim')
parser.add_argument('--chunk_idx', type=int, default=0, help='cdim')

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

def check_completed_jobs(in_fold,jobs):
    uncomplete_jobs = []
    for j in jobs:
        job_dict = load_obj(j,f'{in_fold}/')
        c = experiment_object_gp(job_parameters=job_dict)
        if not os.path.exists(c.save_path+'hyperopt_database.p'):
            uncomplete_jobs.append(j)
    return uncomplete_jobs

def run_func(job_dict):
    c = experiment_object_gp(job_parameters=job_dict)
    c.preprocess_data()
    c.optimize_cg_reg()
    del c
    torch.cuda.empty_cache()

if __name__ == '__main__':
    input  = vars(parser.parse_args())
    fold = input['job_folder']
    chunk_idx = input['chunk_idx']

    jobs = os.listdir(fold)
    jobs = check_completed_jobs(fold,jobs)
    job_chunks_list = np.array_split(jobs, 8)
    job_chunk=job_chunks_list[chunk_idx]

    chunked_input = []
    for el in job_chunk:
        loaded = load_obj(el, folder=f'{fold}/')
        chunked_input.append(loaded)
    for el in chunked_input:
        print(el)
        run_func(el)