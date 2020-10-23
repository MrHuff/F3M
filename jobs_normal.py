import os


def parse_job(l_p,n,min_points,ref_size,a,b,ls,string):
    bash_job = f'./cmake-build-release/cuda_n_body {l_p} {n} {min_points} {ref_size} {a} {b} {ls} 1 ./{string}.csv'
    return bash_job


if __name__ == '__main__':
    for l_p in [4,5,6,7]:
        for n in [100000,1000000,10000000,100000000]:
            for min_points in [250,2500,5000]:
                for a,b in zip([0,100,-100],[1,10,100]):
                    for ls in [0.1,0.5,1,10,200,4000]:
                        bash = parse_job(l_p=l_p,n=n,min_points=min_points,ref_size=5000,a=a,b=b,ls=ls,string='normal_d=3')
                        os.system(bash)