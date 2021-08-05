from experiments import *
import os
def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--par_fac', type=int, nargs='?', default=-1, help='which dataset to run')
    parser.add_argument('--n', type=int, nargs='?', default=-1, help='which dataset to run')
    parser.add_argument('--d', type=int, nargs='?', default=-1, help='which dataset to run')
    parser.add_argument('--index', type=int, nargs='?', default=-1, help='which dataset to run')
    # parser.add_argument('--device', type=int, nargs='?', default=-1, help='which dataset to run')
    return parser

def experiment_1(par_fac,n,d,index):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    par_dict = {1000000:[1000,64],
                10000000: [1000, 64],
                100000000: [5000, 64],
                1000000000: [5000, 64]
    }
    dirname = 'scalability_experiment'
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        for d in [3]:
            node_list = [4]
            for node_nr in node_list:
                nr_of_interpolation = int(node_nr ** d)
                min_points = par_dict[n][0]
                small_field_limit = par_dict[n][1]
                for r2 in [1]:
                    for evarlimit in [0.3]:
                        eff_var_limit = float(evarlimit)
                        # if not os.path.exists(f'{dirname}/{dirname}_{counter}_{par_count}_{n}_{index}.csv'):
                        (X,Y,b) = torch.load(f'uniform_{d}_{n}_{par_fac}.pt')[index]
                        torch.cuda.synchronize()
                        FFM_obj= FFM(X=X,Y=Y, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                                      eff_var_limit=eff_var_limit, var_compression=True,
                                      device="cuda:0", small_field_points=small_field_limit)
                        res_0 = FFM_obj@b


                    #         torch.cuda.synchronize()
                    #         start = time.time()
                    #         res_0 = FFM_obj@b
                    #         end = time.time()
                    #         torch.cuda.synchronize()
                    #         calc_time = end-start
                    #         df = pd.DataFrame([[n,d,nr_of_interpolation,par_count,r2,evarlimit,min_points,calc_time]],columns=['n','d','nr_of_interpolation','par_count','r2','evarlimit','min_points','calc_time'])
                    #         df.to_csv(f'{dirname}/{dirname}_{counter}_{par_count}_{n}_{index}.csv')
                    #         print('Wrote experiments: ',counter)
                    #         del FFM_obj,res_0
                    #         torch.cuda.empty_cache()
                    #     counter+=1
                    #     print('counter: ',counter)
                    # del X,Y,b
                    # torch.cuda.empty_cache()
if __name__ == '__main__':

    input_args = vars(job_parser().parse_args())
    par_fac = input_args['par_fac']
    n = input_args['n']
    d = input_args['d']
    index = input_args['index']
    # dev = input_args['device']
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
    experiment_1(par_fac,n,d,index)