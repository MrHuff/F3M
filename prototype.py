import torch
import time
import numpy as np
# test = torch.randn(int(1e9),device='cuda:0')
# test = torch.randn(int(1e6),3)
# s = time.time()
# boolean = test<0
# e = time.time()
# print(e-s)

def del_none_keys(dict):
    for elem in list(dict):
        if dict[elem] is None:
            del dict[elem]
    return dict
def recursive_center_coordinate(input=[], memory=[]):
    if not input:
        return memory
    else:
        cut = input.pop(0)
        anti_cut = -1*cut
        if not memory:
            memory = [[cut], [anti_cut]]
            return recursive_center_coordinate(input,memory)
        else:
            memory = [el+[cut] for el in memory] + [el+[anti_cut] for el in memory]

            return recursive_center_coordinate(input,memory)

def recursive_divide(input=[], memory=[]):
    if not input:
        return memory
    else:
        cut = input.pop(0)
        anti_cut = ~cut
        if not memory:
            memory = [cut, anti_cut]
            return recursive_divide(input,memory)
        else:
            memory = [el*cut for el in memory]+ [el*anti_cut for el in memory]
            return recursive_divide(input,memory)

def calculate_edge(X,Y):
    Xmin,_ = X.min(dim=0)
    Ymin,_ = Y.min(dim=0)
    Xmax,_ = X.max(dim=0)
    Ymax,_ = Y.max(dim=0)
    comp = torch.cat([Xmax - Xmin, Ymax - Ymin],dim=0)
    edg = torch.max(comp)
    return edg * 1.01,Xmin,Ymin


class n_roon():
    def __init__(self,d,X,X_min,edg):
        self.d = d
        self.current_nr_boxes = 1
        self.data = X
        self.X_min = X_min
        self.edg = edg
        self.centers =  {0: X_min + 0.5*edg}
        self.box_indices = {0: torch.tensor(list(range(X.shape[0])))}
        self.box_nr_points = {0:X.shape[0]}
        self.depth = 0

    def get_mean_points_per_box(self):
        return np.array(list(self.box_nr_points.values())).mean()

    def subdivide(self):
        current_nr_boxes = self.current_nr_boxes
        centers = {i:None for i in range(current_nr_boxes*2**self.d)}
        box_indices = {i:None for i in range(current_nr_boxes*2**self.d)}
        box_nr_points = {i:None for i in range(current_nr_boxes*2**self.d)}
        for i in range(current_nr_boxes):
            current_points = self.data[self.box_indices[i],:]
            cuts = []
            for j in range(self.d):
                cuts.append(current_points[:,j]<=self.centers[i][j])
            divisions = recursive_divide(cuts)
            nr_of_points_boxes = [el.sum() for el in divisions]
            center_coordinates = torch.tensor(recursive_center_coordinate(self.d*[-1]))
            for j in range(2**self.d):
                key = i*2**self.d
                if nr_of_points_boxes[j]>0:
                    box_nr_points[key+j] = nr_of_points_boxes[j].item()
                    centers[key+j] = self.centers[i]+0.25*self.edg*center_coordinates[j,:]
                    box_indices[key+j] = self.box_indices[i][divisions[j]]
        self.box_nr_points = del_none_keys(box_nr_points)
        self.centers = del_none_keys(centers)
        self.box_indices = del_none_keys(box_indices)
        self.current_nr_boxes=len(self.box_nr_points.keys())
        self.edg = self.edg*0.5
        self.depth+=1

if __name__ == '__main__':
    d=3
    X= torch.rand(int(1e4),d)
    Y= torch.rand(int(1e4),d)
    edg,x_min,y_min = calculate_edge(X,Y)
    octaroon_X = n_roon(d=d,X=X,X_min=x_min,edg=edg)
    print(octaroon_X.box_nr_points)
    print(octaroon_X.centers)
    # print(octaroon_X.box_indices)
    octaroon_X.subdivide()
    print(octaroon_X.box_nr_points)
    print(octaroon_X.centers)
    # print(octaroon_X.box_indices)
    octaroon_X.subdivide()

    print(octaroon_X.box_nr_points)
    print(octaroon_X.centers)
    octaroon_X.subdivide()

    print(octaroon_X.box_nr_points)
    print(octaroon_X.centers)
    # print(octaroon_X.box_indices)