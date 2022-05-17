import pickle
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK,Trials,hp, tpe,fmin
import os
from run_gp_experiments_multi_gpu import load_obj,experiment_object_gp
import dill
if __name__ == '__main__':
    job_files = os.listdir('job')
    dicts = []
    for j in job_files:
        dicts.append(load_obj(j,'job/'))
    data = []
    for d in dicts:
        c = experiment_object_gp(d)
        load_data_base_string=c.save_path+ 'hyperopt_database.p'
        trials = pickle.load(open(load_data_base_string, "rb"))
        trials = dill.loads(trials)
        best_trial = sorted(trials.results, key=lambda x: x['test_rsme'], reverse=True)[-1]
        if d['model_string']=='f3m':
            data.append([d['ds_name'],d['model_string'],d['fold'],best_trial['test_rsme'],best_trial['timing'],best_trial['params']['lamb'],best_trial['params']['ls_scale']])
        else:
            data.append([d['ds_name'],d['model_string'],d['fold'],best_trial['test_rsme'],best_trial['timing']*d['nr_of_its'],'N/A','N/A'])
    df = pd.DataFrame(data,columns=['Dataset','Model','Fold','RSME','Training time','lambda','ls_scale'])
    df = df.sort_values(['Dataset','Model','Fold'])
    print(df)
    grp_mean = df.groupby(['Dataset','Model'])['RSME','Training time'].mean().reset_index()
    grp_std = df.groupby(['Dataset','Model'])['RSME','Training time'].std().reset_index()
    print(grp_mean)
    print(grp_std)
