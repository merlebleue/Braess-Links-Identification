from graph_tool.all import *
import os
import pandas as pd
import numpy as np

class Network(Graph):
    """
    Extends the Graph class from graph_tool to work with transportation networks.
    """
    def __init__(self, folder_name: str, *args, **kargs):
        """
        Create a Network from the TPNP github repository. 
        Needs to have the network in "TransportationNetworks"/[folder_name]
        Args:
        - folder_name [str] : name of the network / folder in the github repository
        - args, kargs for the Graph class
        """

        def import_matrix(matfile):
            f = open(matfile, 'r')
            all_rows = f.read()
            blocks = all_rows.split('Origin')[1:]
            matrix = {}
            for k in range(len(blocks)):
                orig = blocks[k].split('\n')
                dests = orig[1:]
                orig=int(orig[0])

                d = [eval('{'+a.replace(';',',').replace(' ','') +'}') for a in dests]
                destinations = {}
                for i in d:
                    destinations = {**destinations, **i}
                matrix[orig] = destinations
            zones = max(matrix.keys())
            mat = np.zeros((zones, zones))
            for i in range(zones):
                for j in range(zones):
                    # We map values to a index i-1, as Numpy is base 0
                    mat[i, j] = matrix.get(i+1,{}).get(j+1,0)

            return mat
        

        folder = os.path.join(os.path.abspath("TransportationNetworks"), folder_name)
        files = {f.split("_")[-1][:-5] : os.path.join(folder, f) for f in os.listdir(folder) if f[-5:]==".tntp"}
        
        # Create the graph
        if "net" in files:
            net = pd.read_csv(files["net"], skiprows=8, sep='\t').drop(['~', ';'], axis=1)
        else:
            raise ValueError("No 'net' file in folder")
        
        if "node" in files:
            nodes = pd.read_csv(files["node"], sep='\t').drop([';'], axis=1)
            super().__init__(len(nodes), *args, **kargs)
            #Add positions
            pos = self.new_vertex_property("vector<float>", nodes[["X", "Y"]].to_numpy())
            self.vertex_properties["pos"] = pos
        else:
            super().__init__(len(net["init_node"].unique()), *args, **kargs)
        
        # Add edge and edge properties
        properties_types = ["long" if i.kind == "i" else "float" for i in net.dtypes]
        edge_properties= list(zip(net.columns, properties_types))[2:]
        net[["init_node", "term_node"]] -= 1 #Start at 0 instead of 1
        self.add_edge_list(list(net.to_records(index=False)), eprops=edge_properties)

        if "trips" in files:
            self.trips = import_matrix(files["trips"])
        if "flow" in files:
            self.TNTP_flows = pd.read_csv(files["flow"], sep='\t')

    def draw(self, interactive=False, flows: EdgePropertyMap =None, **kwargs):
        draw_function = interactive_window if interactive else graph_draw
        
        options = dict(ink_scale=0.5,
                       vertex_text = self.vertex_index)
        if "pos" in self.vp:
            options["pos"] = self.vp.pos

        for k, i in kwargs.items():
            options[k] = i

        if flows != None:
            options["edge_pen_width"] = prop_to_size(flows)
            text = self.new_edge_property("int")
            text.a = np.round(flows.a)
            options["edge_text"] = text

        return draw_function(self, **options)
    
    def get_edge_mask(self, edges_list: np.array):
        edge_array = self.get_edges()
        
        mask = (edge_array[:,None] == edges_list).all(2).any(1)
        return mask