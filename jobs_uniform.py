import os


def parse_job(l_p,n,min_points,ref_size,a,b,ls,string):
    bash_job = f'./cmake-build-release/cuda_n_body {l_p} {n} {min_points} {ref_size} {a} {b} {ls} 1 ./{string}.csv'
    return bash_job


if __name__ == '__main__':
    for l_p in [4,5,6,7]:
        for n in [1000000000]:
            if n==100000:
                min_points_list = [500,1000]
            elif n==1000000:
                min_points_list = [500,1000,5000]
            elif n==10000000:
                min_points_list = [1000,10000]
            elif n==100000000:
                min_points_list = [1000,10000,25000]
            elif n==1000000000:
                min_points_list = [10000,25000,100000]
            for min_points in min_points_list:
                for a,b in zip([0,-1,-100],[1,1,100]):
                    for ls in [0.1,0.5,1,10,200,4000]:
                        bash = parse_job(l_p=l_p,n=n,min_points=min_points,ref_size=5000,a=a,b=b,ls=ls,string='uniform_d=3')
                        os.system(bash)