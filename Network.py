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

        self.folder_name = folder_name

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
        
        # Add edges and edge properties
        properties_types = ["long" if i.kind == "i" else "float" for i in net.dtypes]
        edge_properties= list(zip(net.columns, properties_types))[2:]
        net[["init_node", "term_node"]] -= 1 #Start at 0 instead of 1
        self.add_edge_list(list(net.to_records(index=False)), eprops=edge_properties)

        if "trips" in files:
            self.trips = import_matrix(files["trips"])
        if "flow" in files:
            self.TNTP_flows = pd.read_csv(files["flow"], sep='\t')

    def draw(self, interactive=False, flows: EdgePropertyMap =None, flows_by_o = None, flows_by_OD = None, o=None, d=None, **kwargs):
        draw_function = interactive_window if interactive else graph_draw
        
        options = dict(ink_scale=0.5,
                       vertex_text = self.vertex_index)
        if "pos" in self.vp:
            options["pos"] = self.vp.pos

        for k, i in kwargs.items():
            options[k] = i

        if isinstance(flows_by_OD, np.ndarray):
            if d == None:
                flows_by_OD = flows_by_OD.sum(axis=1)
            else:
                flows_by_OD = flows_by_OD[:, d, :]
            if o == None:
                flows_by_OD = flows_by_OD.sum(axis=0)
            else:
                flows_by_OD=flows_by_OD[o]

            flows = self.new_edge_property("float", vals = flows_by_OD)

        if flows_by_o != None:
            if type(o) == int:
                flows = self.new_edge_property("float", vals = flows_by_o.get_2d_array()[o,:])
            elif o == "sum":
                flows = self.new_edge_property("float", vals = flows_by_o.get_2d_array().sum(axis=0))

        if flows != None:
            options["edge_pen_width"] = prop_to_size(flows)
            text = self.new_edge_property("int")
            text.a = np.round(flows.a)
            options["edge_text"] = text
            options["edge_text_color"] = self.new_edge_property("string", vals=np.where(flows.a > flows.a.max()/1000, "black", "grey"))

        return draw_function(self, **options)
    
    def get_edge_mask(self, edges_list: np.array):
        edge_array = self.get_edges()
        
        mask = (edge_array[:,None] == edges_list).all(2).any(1)
        return mask
    
    def save_flow(self, flow: EdgePropertyMap | np.ndarray, name: str, folder = "files"):
        """
        Saves the flow (either total flows or flows by origin) in a txt file, with columns `s, t, v1[, v2, ..]`
        """
        if isinstance(flow, np.ndarray):
            # 3d array
            dim=3

            # Get the shape of the 3D array
            n_rows, n_cols, n_depth = flow.shape

            # Generate indices j and k
            j, k = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')

            # Flatten the arrays to iterate easily
            j = j.flatten()
            k = k.flatten()
            values = flow[j, k]

            # Stack j, k, and the values along the second axis to create the 2D array
            array = np.hstack((j[:, np.newaxis], k[:, np.newaxis], values))
        elif "vector" in flow.value_type():
            # We need to take the 2d array
            dim=2
            array = flow.get_2d_array().T
            array = np.hstack((self.get_edges(), array))
        else:
            dim = 1
            array = flow.get_array().reshape((-1, 1))
            array = np.hstack((self.get_edges(), array))
        
        np.savetxt(os.path.join(folder, "_".join([self.folder_name, str(dim) + "D", name])), array)

    def load_flow(self, name: str, dim = None, folder = "files"):
        """
        Load a flow (either total flows or flows by origin) from a txt file, previously saved with the .save_flow method, with columns `s, t, v1[, v2, ..]`
        """
        list_of_files = os.listdir(folder)
        candidates = [f for f in list_of_files if f.split("_")[0] == self.folder_name and "_".join(f.split("_")[2:]) == name]
        
        match len(candidates):
            case 0:
                print([(f.split("_")[0], "_".join(f.split("_")[2:])) for f in list_of_files])
                raise ValueError(f"No file found with network {self.folder_name} and name {name} in folder {folder}. Found files : {list_of_files}")
            case 1:
                file = candidates[0]
            case _:
                if dim == None:
                    raise ValueError(f"More than one file found with network {self.folder_name} and name {name} in folder {folder}. Specify dimension. Found files : {list_of_files}")
                else:
                    c = [c for c in candidates if c.split("_")[1][0] == dim]
                    if len(c) != 1:
                        raise ValueError(f"No file or more than one file found with the required dimension. Corresponding files found : {candidates}")
                    else :
                        file = c[0]
        
        array = np.loadtxt(os.path.join(folder, file))
        dim = int(file.split("_")[1][0])
        if dim==3:
            # 3D array encoded as 2D array

            # Extract j, k, and values
            j = array[:, 0].astype(int)
            k = array[:, 1].astype(int)
            values = array[:, 2:]

            # Determine the shape of the original 3D array
            n_rows = j.max() + 1
            n_cols = k.max() + 1
            n_depth = values.shape[1]

            # Reconstruct the 3D array
            array = np.zeros((n_rows, n_cols, n_depth))
            array[j, k] = values
            return array

        if (array [:, :2] == self.get_edges()).all():
            match dim:
                case 1:
                    return self.new_edge_property("float", vals = array[:, -1])
                case 2 :
                    return self.new_edge_property("vector<float>", vals = array[:, 2:])
                case _ :
                    raise ValueError(f"Error determining dimension. Found dim={dim} for file {file}")
        else:
            match dim:
                case 1 :
                    prop = self.new_edge_property("float")
                case 2 :
                    prop = self.new_edge_property("vector<float>")
                case _:
                    raise ValueError(f"Error determining dimension. Found dim={dim} for file {file}")
            
            for e in self.get_edges():
                value = array[array[:, :2]==e, 2:]
                #if dim==1:
                #    value = value[0]
                prop[e] = value

            return prop
        
    def export_flow(self, flow: EdgePropertyMap | np.ndarray, name: str, folder = "exports", OD_demand=None):
        """
        Exports the flow (either total flows or flows by origin) in a formatted txt file, with columns `s, t, v1[, v2, ..]`
        """
        if isinstance(flow, np.ndarray):
            # 3d array
            dim=3

            # Get the shape of the 3D array
            n_rows, n_cols, n_depth = flow.shape

            # Generate indices j and k
            j, k = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')

            # Flatten the arrays to iterate easily
            j = j.flatten()
            k = k.flatten()
            values = flow[j, k]

            # Stack j, k, and the values along the second axis to create the 2D array
            array = np.hstack((j[:, np.newaxis], k[:, np.newaxis], values))
            if OD_demand is None:
                df = pd.DataFrame(array[:, 2:], columns=pd.MultiIndex.from_tuples([(i,j) for i,j in self.get_edges()], names = ["i", "j"]), index=pd.MultiIndex.from_arrays([array[:, 0].astype("int"), array[:, 1].astype("int")], names = ["O", "D"])).T
                df = df.rename(columns="{: >8}".format)
            else:
                df = pd.DataFrame(array[:, 2:], columns=pd.MultiIndex.from_tuples([(i,j) for i,j in self.get_edges()], names = ["i", "j"]), index=pd.MultiIndex.from_tuples([(f"{o:^3}->{d:3}", f"{OD_demand[o,d]:8.2f}") for o, d in array[:, :2].astype("int")], names = ["o>d", "tot"])).T
        elif "vector" in flow.value_type():
            # We need to take the 2d array
            dim=2
            array = flow.get_2d_array().T
            #array = np.hstack((self.get_edges(), array))
            df = pd.DataFrame(array, columns = [f"{r: >8}" for r in self.get_vertices()], index=pd.MultiIndex.from_arrays(self.get_edges().T.tolist(), names = ["i", "j"]))
            df.columns.name = "o"
        else:
            dim = 1
            array = flow.get_array()
            #array = np.hstack((self.get_edges(), array))
            df = pd.Series(array, name = name, index=pd.MultiIndex.from_arrays(self.get_edges().T.tolist(), names = ["i", "j"]))
        
        df.to_csv(os.path.join(folder, "_".join([self.folder_name, str(dim) + "D", name])), sep="\t", float_format="%8.2f")

    def export_paths(self, paths, OD_demand, name="paths", folder="exports"):
        str = ""
        for (r,s), paths_list in paths.items():
            str += f"{r:3d} -> {s:3d} ({OD_demand[r, s]}):\n"
            for path, flow in paths_list:
                    str += f"\t{flow:8.3g} : {path}\n"
            str += "\n"

        with open(os.path.join(folder, "_".join([self.folder_name, name]) + ".txt"), "w") as f:
                  f.write(str)
        return str
    