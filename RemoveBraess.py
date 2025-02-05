from Network import Network
from FrankWolf import *
from EntropyMaximisation import *

import graph_tool as gt
import numpy as np 
import pandas as pd
from tqdm import tqdm
import os

def compute_total_travel_time(net: Network, flows_array: np.ndarray, cost_function = BTR_cost_function):
    return (cost_function(flows_array, net).a * flows_array).sum()

def try_removing_braess(
        net: Network,
        mask: np.ndarray,
        save_folder: str = "files/masked",
        export_folder: str = "exports/masked",
        name_pattern = "o{o}d{d}"
):
    
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(export_folder, exist_ok=True)

    for o, d in tqdm(list(zip(*np.where(mask.sum(axis=2)>0)))):
        masked_by_origin, masked_flows = frankwolf(net, net.trips, OD_mask={(o,d) : ~mask[o,d]}, verbose=0, tolerance= 1e-4, n_max=2e5)
        
        # Save and export the flows
        net.save_flow(masked_by_origin, name_pattern.format(o=o, d=d), folder=save_folder)
        net.save_flow(masked_flows, name_pattern.format(o=o, d=d), folder=save_folder)
        net.export_flow(masked_by_origin, name_pattern.format(o=o, d=d), folder=export_folder)
        net.export_flow(masked_flows, name_pattern.format(o=o, d=d), folder=export_folder)

def remove_all_OD(net: Network,
                  cases: list,
                  save_folder: str = "files/masked",
                  export_folder: str = "exports/masked",
                  name_pattern = "n{n}_l{l}"):

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(export_folder, exist_ok=True)

    ODs = list(zip(*np.where(net.trips > 0)))

    for edges_indices in tqdm(cases):
        if type(edges_indices) is int:
            edges_indices = [edges_indices]
        n = len(edges_indices)
        l = "_".join([str(l) for l in edges_indices])
        try:
            masked_by_origin = net.load_flow(name_pattern.format(n=n, l=l), dim=2,folder=save_folder, verbose=0)
            masked_flows  = net.load_flow(name_pattern.format(n=n, l=l), dim=1,folder=save_folder, verbose=0)
        except ValueError:        
            mask = ~np.isin(net.edge_index, edges_indices)

            try:
                masked_by_origin, masked_flows = frankwolf(net, net.trips, OD_mask={od: mask for od in ODs}, verbose=1, tolerance= 1e-4, n_max=2e5)
            
                # Save and export the flows
                net.save_flow(masked_by_origin, name_pattern.format(n = n, l=l), folder=save_folder)
                net.save_flow(masked_flows, name_pattern.format(n = n, l=l), folder=save_folder)
                net.export_flow(masked_by_origin, name_pattern.format(n = n, l=l), folder=export_folder)
                net.export_flow(masked_flows, name_pattern.format(n = n, l=l), folder=export_folder)
            except OverflowError:
                print(f"Error for links {[l+1 for l in edges_indices]}")