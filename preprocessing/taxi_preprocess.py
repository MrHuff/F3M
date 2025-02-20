import pandas as pd
import torch
import os
# os.environ["MODIN_ENGINE"] = "dask"

from azureml.opendatasets import NycTlcYellow
import numpy as np
from datetime import datetime
from dateutil import parser
import time
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import torch
class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])
def reject_outliers(data,mean,std, m=2.):
    mask = ((data - mean).abs() >= m * std).any(1)
    return data[~mask]


class Kernel(torch.nn.Module):
    def __init__(self,):
        super(Kernel, self).__init__()

    def sq_dist(self,x1,x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))
        # Zero out negative values
        res.clamp_min_(0)

        # res  = torch.cdist(x1,x2,p=2)
        # Zero out negative values
        # res.clamp_min_(0)
        return res

    def covar_dist(self, x1, x2):
        return self.sq_dist(x1,x2).sqrt()

    def get_median_ls(self, X,Y=None):
        with torch.no_grad():
            if Y is None:
                d = self.covar_dist(x1=X, x2=X)
            else:
                d = self.covar_dist(x1=X, x2=Y)
            ret = torch.sqrt(torch.median(d[d >= 0])) # print this value, should be increasing with d
            if ret.item()==0:
                ret = torch.tensor(1.0)
            return ret

ref_kernel = Kernel()
if __name__ == '__main__':
    if not os.path.exists('taxi.parquet'):
        start_date = parser.parse('2009-01-01')
        end_date = parser.parse('2018-12-31')
        nyc_tlc = NycTlcYellow(start_date=start_date, end_date=end_date,cols=['tpepPickupDateTime','tpepDropoffDateTime','fareAmount','tripDistance','tipAmount'])
        nyc_tlc_df = nyc_tlc.to_pandas_dataframe()
        nyc_tlc_df.to_parquet('taxi.parquet')
    if not os.path.exists('taxi.pt'):
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=True)

        # from distributed import Client
        # client = Client()
        # import modin.pandas as pd
        from tqdm import tqdm
        # from modin.config import ProgressBar
        # ProgressBar.enable()
        # import ray
        # ray.init()
        start = time.time()
        nyc_tlc_df = pd.read_parquet('taxi.parquet')
        # with ProgressBar():
        #     nyc_tlc_df = dd.read_parquet('taxi.parquet').compute()
        end = time.time()
        print(end-start)
        nyc_tlc_df['trip_time'] = nyc_tlc_df['tpepDropoffDateTime'] - nyc_tlc_df['tpepPickupDateTime']
        end_2 = time.time()
        print(end_2-end)
        nyc_tlc_df['trip_time'] = nyc_tlc_df['trip_time'].parallel_apply(lambda x: x.total_seconds()/60)
        end_3 = time.time()
        print(end_3-end)
        nyc_tlc_df = nyc_tlc_df.drop(['tpepDropoffDateTime','tpepPickupDateTime'],axis=1)
        print(nyc_tlc_df.columns)
        data = torch.from_numpy(nyc_tlc_df.values).float()
        torch.save(data,'taxi.pt')

    if not os.path.exists('cleaned_taxi.pt'):
        tensor = torch.load('taxi.pt')
        print(tensor.shape)
        tensor = tensor[~torch.any(tensor.isnan(), dim=1)]
        print(tensor.shape)
        mean = tensor.mean(0)
        std = tensor.std(0)
        tensor = reject_outliers(tensor,mean,std,0.025)
        print(tensor.shape)
        mean = tensor.mean(0)
        std = tensor.std(0)
        print(mean)
        print(std)
        torch.save(tensor,'cleaned_taxi.pt')
    if not os.path.exists('krr_taxi.pt'):
        #['fareAmount', 'tipAmount', 'tripDistance', 'trip_time']
        tensor = torch.load('cleaned_taxi.pt')
        tensor = tensor[:1000000000,:]
        X_2 = tensor[:,[0,2,3]]
        y_2 = tensor[:,1].unsqueeze(-1)
        y_2 =torch.where(y_2 > 0, 1, -1).float()


        mean = X_2.mean(0)
        std = X_2.std(0)
        print(mean)
        print(std)
        X_2 -=mean
        X_2/=std

        perm = torch.randperm(X_2.size(0))
        idx = perm[:10000]
        samples = X_2[idx]
        ls = ref_kernel.get_median_ls(samples)
        print(ls)
        data_dict_2 = {'X': X_2.float(), 'y': y_2.float(), 'ls': ls.float().item()}
        torch.save(data_dict_2, f'krr_taxi.pt')

    if not os.path.exists('taxi_debug.pt'):
        data=torch.load('krr_taxi.pt')
        my_values = {
            'X':data['X']
        }

        # Save arbitrary values supported by TorchScript
        # https://pytorch.org/docs/master/jit.html#supported-type
        container = torch.jit.script(Container(my_values))
        container.save("taxi_debug.pt")

    # data = pd.read_parquet('osm-1billion.parq')
    # data = torch.from_numpy(data.values)
    # if not os.path.exists('raw_data_osm.pt'):
    #     torch.save(data,'raw_data_osm.pt')
    # if not os.path.exists('standardized_data_osm.pt'):
    #     mean = data.mean(0)
    #     std = data.std(0)
    #     data -=mean
    #     data/=std
    #     torch.save(data,'standardized_data_osm.pt')
    #     print(data.var(0))
    # if not os.path.exists('minmax_data_osm.pt'):
    #     max,_ = data.max(0)
    #     min,_ = data.min(0)
    #     data -=min
    #     data/=(max-min)
    #     torch.save(data,'minmax_data_osm.pt')
    #     print(data.var(0))



    # torch.manual_seed(eff_var)
    # np.random.seed(eff_var)
    # X_3 = torch.load('standardized_data_osm.pt')
    # X_3 = X_3[:N]
    # arr = np.random.randint(0, N, (5000, 1))
    # arr = np.unique(arr)
    # alpha = torch.randn(5000, 1)  # Had to adjust this, else FALKON fails to converge at all
    # x_ref_3 = X_3[arr[:5000], :]
    # ls = torch.cdist(x_ref_3, x_ref_3).median().item()
    # object = benchmark_matmul(X_3, x_ref_3, ls=ls)
    # y_3 = object @ alpha
    # X_3 = X_3.cpu()
    # y_3 = y_3.cpu()
    # data_dict_3 = {'X': X_3, 'y': y_3, 'ls': ls, 'eff_var': ls}
    # # torch.save(data_dict_3,f'real_problem_N={N}_eff_var={eff_var}_5.pt')
    # torch.save(data_dict_3, f'real_problem_N={N}_seed={eff_var}.pt')
    # print(y_3[:100])



