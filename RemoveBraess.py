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

def try_removing_braess_1_OD(
        net: Network,
        UE_OD_flows: np.ndarray,
        SO_OD_flows: np.ndarray,
        save_folder: str = "files/masked",
        export_folder: str = "exports/masked",
        name_pattern = "o{o}d{d}n1l{l}",
        masking_rule = lambda UE_OD_flows, SO_OD_flows :((UE_OD_flows > 0.1) & (SO_OD_flows < 0.1))
):
    mask = masking_rule(UE_OD_flows, SO_OD_flows)

    df = pd.DataFrame(columns=["o", "d", "l", "N pot Braess links", "Total Travel Time"], dtype="float")
    
    previous_ttt = compute_total_travel_time(net, UE_OD_flows.sum(axis=(0, 1)))
    for o, d in (pbar := tqdm(list(zip(*np.where(mask.sum(axis=2)>0))))):
        pbar.set_postfix({"OD": (o, d)})
        for l in (pbar2 := tqdm(np.where(mask[o, d])[0], leave=False)):
            pbar2.set_postfix({"l": l})
            mask_l = np.full(net.num_edges(), True)
            mask_l[l] = False
            masked_by_origin, masked_flows = frankwolf(net, net.trips, OD_mask={(o,d) : mask_l}, verbose=0, tolerance= 1e-4, n_max=2e5)
            
            # Save and export the flows
            net.save_flow(masked_by_origin, name_pattern.format(o=o, d=d, l=l), folder=save_folder)
            net.save_flow(masked_flows, name_pattern.format(o=o, d=d, l=l), folder=save_folder)
            net.export_flow(masked_by_origin, name_pattern.format(o=o, d=d, l=l), folder=export_folder)
            net.export_flow(masked_flows, name_pattern.format(o=o, d=d, l=l), folder=export_folder)

            # Add a row to the df

            ttt= compute_total_travel_time(net, masked_flows.a)
            df = [
                o,d,l,
                mask[o,d].sum(),
                ttt
            ]

            print(o, d, l, mask[o,d].sum(), ttt, previous_ttt - ttt)

    #Write csv
    df.set_index(["o", "d", "l"])
    df["Improvement"] = previous_ttt - df["Total Travel Time"] 


    df[["Total Travel Time", "Improvement"]] = df[["Total Travel Time", "Improvement"]].astype("float")
    df.to_csv(os.path.join(export_folder, "_".join([net.folder_name, "comparison_1link"])), sep="\t", float_format="%8.2f")

    return df

def remove_1link_1OD(
        net: Network,
        OD_flow: np.ndarray,
        save_folder: str = "files/masked",
        export_folder: str = "exports/masked",
        name_pattern = "o{o}d{d}l{l}",
):
    
    for o, d, l in tqdm(list(zip(*np.where(OD_flow>0)))):
        mask_l = np.full(net.num_edges(), True)
        mask_l[l] = False
        masked_by_origin, masked_flows = frankwolf(net, net.trips, OD_mask={(o,d) : mask_l}, verbose=0, tolerance= 1e-4, n_max=2e5)

        # Save and export the flows
        net.save_flow(masked_by_origin, name_pattern.format(o=o, d=d, l=l), folder=save_folder)
        net.save_flow(masked_flows, name_pattern.format(o=o, d=d, l=l), folder=save_folder)
        net.export_flow(masked_by_origin, name_pattern.format(o=o, d=d, l=l), folder=export_folder)
        net.export_flow(masked_flows, name_pattern.format(o=o, d=d, l=l), folder=export_folder)

    
