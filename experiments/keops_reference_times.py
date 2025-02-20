from experiments import *

def experiment_1(device="cuda:0",d=3):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    dirname = 'keops_ref_bench'
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        # for n in [1000000, 10000000, 100000000]:
        for n in [1000000]:
            torch.manual_seed(seed)
            X = torch.empty(n, d).uniform_(0, (12) ** 0.5)
            b = torch.empty(n, 1).normal_(0, 1)

            keops_benchmark_0 = keops_matmul(X, X, ls=ls, device=device)  # get some references
            start = time.time()
            true_0 = keops_benchmark_0 @ b  # calculate reference
            end = time.time()
            torch.cuda.synchronize()
            calc_time = end - start
            torch.cuda.empty_cache()
            print("benchmarks done\n")
            if not os.path.exists(f'{dirname}/{dirname}_{counter}_{d}.csv'):
                df = pd.DataFrame([[n,d,calc_time]],columns=['n','d','calc_time'])
                df.to_csv(f'{dirname}/{dirname}_{counter}_{d}.csv')
                print('Wrote experiments: ',counter)
                counter+=1
                print('counter: ',counter)

if __name__ == '__main__':
    input_args = vars(job_parser().parse_args())
    idx = input_args['idx']
    if idx == 1:
        experiment_1(d=3)
    elif idx == 2:
        experiment_1(d=4)
    elif idx == 3:
        experiment_1(d=5)
