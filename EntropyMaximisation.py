from Network import Network
from FrankWolf import *

import graph_tool as gt
import numpy as np 
from tqdm.notebook import tqdm

def backward_entropy_maximisation(net: Network, flows_by_o: gt.EdgePropertyMap, n:int):
    """
    TODO :
    - Check condition of n
    - Change set into list to control the order ?
    """
    
    # Variable Initialisation
    x_n = net.new_edge_property("float")
    residuals = flows_by_o.copy()

    x_a = flows_by_o.get_2d_array()
    x_n_a = x_n.a
    residuals_a = residuals.get_2d_array()

    n_nodes = x_a.shape[0]
    n_edges = x_a.shape[1]

    # Initialise
    eta_rj = np.zeros((n_nodes, n_nodes))
    psi_r_ij = np.zeros(x_a.shape)
    for j in net.iter_vertices():
        in_edge_indices = net.get_in_edges(j, [net.edge_index])[:, -1]
        eta_rj[:, j] = x_a[:, in_edge_indices].sum(axis=1)
        psi_r_ij[:, in_edge_indices] = np.where(eta_rj[:,[j]] > 0, x_a[:, in_edge_indices]/eta_rj[:,[j]], 0)

    # Main loop
    for r in range(n_nodes): # For each origin
        if eta_rj[r, n] > 0:
            K = [n]
            k_r = np.zeros((n_nodes))
            k_r[n] = eta_rj[r, n]
            while len(K) > 0:
                m = K.pop(0)
                if k_r[m] > 0:
                    omega = k_r[m]
                    k_r[m] = 0
                    i, _, e = net.get_in_edges(m, [net.edge_index]).T
                    i, e = i[psi_r_ij[r,e]>0], e[psi_r_ij[r,e]>0]
                    alpha = omega * psi_r_ij[r, e]
                    residuals_a[r, e] -= alpha
                    x_n_a[e] += alpha
                    K.extend(i[i!=r])
                    K = list(dict.fromkeys(K)) #remove duplicates
                    k_r[i[i!=r]] += alpha[i!=r]

    residuals.set_2d_array(residuals_a)
    x_n.a = x_n_a

    return x_n, residuals

def forward_entropy_maximization(net: Network, x_n: gt.EdgePropertyMap, residuals: gt.EdgePropertyMap, n: int):
    new_flows = residuals.copy()
    new_flows_a = new_flows.get_2d_array()
    x_n_a = x_n.copy().a
    
    n_nodes = new_flows_a.shape[0]
    n_edges = new_flows_a.shape[1]

    #Initialisation
    theta = np.zeros(n_nodes)
    phi_n = np.zeros(n_edges)
    for i in net.iter_vertices():
        out_edge_indices = net.get_out_edges(i, [net.edge_index])[:, -1]
        theta[i] = x_n_a[out_edge_indices].sum()
        if theta[i] > 0:
            phi_n[out_edge_indices] = x_n_a[out_edge_indices]/theta[i]
        else:
            phi_n[out_edge_indices] = 0
    
    # Main loop
    for r in range(n_nodes):
        out_edge_indices = net.get_out_edges(r, [net.edge_index])[:, -1]
        in_edge_indices = net.get_in_edges(r, [net.edge_index])[:, -1]
        d_rn = x_n_a[out_edge_indices].sum() - x_n_a[in_edge_indices].sum()
        
        if d_rn > 0:
            K = {r}
            k_n = np.zeros((n_nodes))
            k_n[r] = d_rn
            while len(K) > 0:
                m = K.pop()
                if k_n[m] > 0:
                    omega = k_n[m]
                    k_n[m] = 0
                    _, j, e = net.get_out_edges(m, [net.edge_index]).T
                    j, e = j[x_n_a[e] > 0], e[x_n_a[e]>0]
                    alpha = omega * phi_n[e]
                    new_flows_a[r, e] += alpha
                    x_n_a[e] -= alpha
                    K.update(j[j != n])
                    k_n[j[j != n]] += alpha[j != n]

    new_flows.set_2d_array(new_flows_a)

    return new_flows

def EMARB(net: Network,
          flows_by_o: gt.EdgePropertyMap,
          epsilon: float,
          sigma: float = 0.5,
          M: int = 5):
    """
    Applies the Entropy Maximization through Alternative Representation of Bushes (EMARB) algorithm from Xie and Nie : A New Algorithm for Proportionality

    Args :
    - net (from class Network): the network on which to run the algorithm
    - flows_by_o (from graph_tool.EdgePropertyMap): A feasible origin-based link flow solution that satisfies the UE condition
    - epsilon : the convergence criterion
    - sigma [<=1] : node update threshold
    - M : number of iterations between which all nodes are visited
    """

    # Pre-processing
    flows = flows_by_o.copy()
    flows_a = flows.get_2d_array()
    n_nodes, n_edges = flows_a.shape
    
    # Initialize
    chi = np.full(n_nodes, np.inf)
    chi_barre = 0

    # Main Loop
    def generator():
        n_iter = 0
        while not (chi<epsilon).all():
            n_iter += 1
            yield n_iter
    for iter in tqdm(generator()) :
        for n in tqdm(range(n_nodes), desc = f"{iter}: {".".join(np.where(chi>sigma*chi_barre, "X", " "))}"):
            if chi[n]>sigma*chi_barre or iter%M==0 :
                x_n, residuals = backward_entropy_maximisation(net, flows, n)
                flows = forward_entropy_maximization(net, x_n, residuals, n)
                chi[n] = np.abs(flows.get_2d_array() - flows_a).sum()
                flows_a = flows.get_2d_array()
        chi_barre = chi.sum() / n_nodes
    
    return flows

def get_OD_flows(net: Network, OD: np.array, flows_by_origin: gt.EdgePropertyMap, return_paths = True, flow_limit: float = 1e-14):
    # Translate the OD array in a DO_pairs dict of sets {Dest : {Origs}} if flow > 0 :
    DO_pairs_dict = {}
    DO_pairs = set()
    for s in OD.sum(axis=0).nonzero()[0]:
        DO_pairs_dict[s] = set(OD[:, s].nonzero()[0])
        DO_pairs.update({(r, s) for r in OD[:, s].nonzero()[0]})

    # Prepare to compute the flows
    x_a = flows_by_origin.get_2d_array()
    n_nodes = x_a.shape[0]
    eta_rj = np.zeros((n_nodes, n_nodes))
    psi_r_ij = np.zeros(x_a.shape)
    for j in net.iter_vertices():
        in_edge_indices = net.get_in_edges(j, [net.edge_index])[:, -1]
        eta_rj[:, j] = x_a[:, in_edge_indices].sum(axis=1)
        psi_r_ij[:, in_edge_indices] = np.where(eta_rj[:,[j]] > 0, x_a[:, in_edge_indices]/eta_rj[:,[j]], 0)
    
    # Create paths and OD flows variables
    if return_paths:
        paths = {(r,s): []  for s, sources in DO_pairs_dict.items() for r in sources}
    OD_flows = np.zeros((*OD.shape, net.num_edges()), dtype=np.float64) # 3D array for O, D, link flow values


    # Prepare alg. 4 to get the paths
    def DFSPath(k: int):
        for r in net.get_in_neighbors(k):
            if r in Q :
                #Avoid looping
                pass
            else:
                edges.append((r, Q[-1]))
                Q.append(r)
                if r in sources:
                    # Found a path to an origin

                    # Compute the flows
                    mask = net.get_edge_mask(edges)
                    flow = OD[r,s] * psi_r_ij[r, mask].prod()

                    # Save it in the paths variable
                    if flow > flow_limit and return_paths:
                        paths[(r,s)].append((Q.copy(), flow))

                    # Save it in the OD_flows array
                    if flow > flow_limit:
                        OD_flows[r,s,mask] += flow

                DFSPath(r)
                Q.remove(r)
                edges.pop()

    # Run the algorithm
    for s, sources in tqdm(DO_pairs_dict.items()):
        Q = [s]
        edges = []
        DFSPath(s)

    # Return OD_flows and paths if asked
    if return_paths:
        return OD_flows, paths
    
    # Return OD_flows otherwise
    return OD_flows

            
            
    
            

    
                

