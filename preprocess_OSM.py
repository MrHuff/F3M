import pandas as pd
import torch
import os
class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])
if __name__ == '__main__':
    data = pd.read_parquet('osm-1billion.parq')
    data = torch.from_numpy(data.values)
    # if not os.path.exists('raw_data_osm.pt'):
    #     torch.save(data,'raw_data_osm.pt')
    if not os.path.exists('standardized_data_osm.pt'):
        mean = data.mean(0)
        std = data.std(0)
        data -=mean
        data/=std
        torch.save(data,'standardized_data_osm.pt')
        print(data.var(0))
    if not os.path.exists('osm_debug.pt'):
        X=torch.load('standardized_data_osm.pt')
        X = X[:1000000,:]
        my_values = {
            'X':X.float()
        }

        # Save arbitrary values supported by TorchScript
        # https://pytorch.org/docs/master/jit.html#supported-type
        container = torch.jit.script(Container(my_values))
        container.save("osm_debug.pt")
    # if not os.path.exists('minmax_data_osm.pt'):
    #     max,_ = data.max(0)
    #     min,_ = data.min(0)
    #     data -=min
    #     data/=(max-min)
    #     torch.save(data,'minmax_data_osm.pt')
    #     print(data.var(0))





