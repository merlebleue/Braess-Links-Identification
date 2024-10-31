from Network import Network
import graph_tool as gt
import numpy as np 
from scipy.optimize import bisect
from tqdm.notebook import tqdm

def shortest_path(net: Network, times: gt.EdgePropertyMap, origin: int, n_max=1e5):
    labels = net.new_vertex_property("float", np.inf)
    predecessor = net.new_vertex_property("int", -1)
    sequence = {origin}
    labels[origin] = 0
    edges = net.get_edges([times])

    n_ite = 0
    while len(sequence) > 0 and n_ite < n_max:
        n_ite += 1
        # Take a node from the sequence set
        i = sequence.pop() 

        # Get the edges and neighbors from this node
        edges_of_i = edges[edges[:, 0] == i] 
        neighbors = edges_of_i[:, 1].astype("int")

        # Get the nodes that are improved and change the label and predecessor
        mask = (labels.a[neighbors] - labels[i] > edges_of_i[:,2])
        improved_nodes = neighbors[mask]
        labels.a[improved_nodes] = labels[i] + edges_of_i[:,2][mask]
        predecessor.a[improved_nodes] = i
        
        # Update the sequence set
        sequence.update(improved_nodes)

    # Get a predecessors and links list for each destination:
    predecessors = {}
    links = {}
    for j,i in net.iter_vertices([predecessor]):
        if j == origin:
            pass
        elif i in predecessors :
            predecessors[j] = predecessors[i] + [i]
            links[j] = links[i] + [[i, j]]
        else :
            predecessors[j] = [i]
            links[j] = [[i, j]]
            while i != origin :
                links[j] += [[predecessor[i], i]]
                i = predecessor[i]
                predecessors[j] = [i] + predecessors[j]
            
    return links, predecessors

def BTR_cost_function(flows_array: np.array, net: Network):
    computed_times = net.new_edge_property("float")
    get = lambda s : net.ep[s].a
    computed_times.a = get("free_flow_time") * (1 + get("b") * (flows_array/get("capacity"))**get("power"))
    return computed_times

def frankwolf(net: Network, OD : np.array, shortest_path_alg = shortest_path, cost_function = BTR_cost_function, n_max=1e5, tolerance=1e-3, verbose=0):
    
    def direction_search(times : gt.EdgePropertyMap):
        """
            Performs all-or-nothing assignment based on the 'times' edge property
            Returns :
            - computed_flows : an EdgePropertyMap, which countains for each edge a vector with the flow from each origin
        """
        computed_flows = net.new_edge_property("vector<float>", vals=np.zeros((net.num_edges(), net.num_vertices())))
        flows_array = computed_flows.get_2d_array()

        # For each origin
        for o in range(OD.shape[0]):
            # Use shortest path algorithm to get list of links for each destination
            paths,_ = shortest_path_alg(net, times, o)
            # Convert list of links to numpy mask
            paths_mask = {k : net.get_edge_mask(i) for k,i in paths.items()}
            # Add flow of each destination to each link in its path
            for d, edge_mask in paths_mask.items():
                flows_array[o, edge_mask] += OD[o, d]
        
        # Set the array to the edge property
        computed_flows.set_2d_array(flows_array)

        return computed_flows
    
    def get_z_prime(alpha: float, flows: gt.EdgePropertyMap, direction: gt.EdgePropertyMap):
        times = BTR_cost_function(flows.a + alpha*(direction.a - flows.a), net)
        return ((direction.a - flows.a) * times.a).sum()

    # Initialisation
    flows_by_o = net.new_edge_property("vector<float>", vals=np.zeros((net.num_edges(), net.num_vertices())))
    total_flows = net.new_edge_property("float", vals=flows_by_o.get_2d_array().sum(axis=0))

    times = BTR_cost_function(total_flows.a, net)
    prev_total_time = 1e10 
    total_time = times.a.sum()

    flows_by_o = direction_search(times)
    total_flows.a = flows_by_o.get_2d_array().sum(axis=0)

    #Loop
    n_iter = 0
    def generator():
        while abs(total_time - prev_total_time) > tolerance and n_iter < n_max:
            yield
    if verbose >0: #For following the progress
        generator = tqdm(generator())
    else :
        generator = generator()
    for _ in generator:
        n_iter = n_iter + 1


        # Update time
        times = BTR_cost_function(total_flows.a, net)
        prev_total_time = total_time
        total_time = times.a.sum()

        # Direction search
        direction_by_o = direction_search(times)
        direction = net.new_edge_property("float", vals=direction_by_o.get_2d_array().sum(axis=0))

        # Line search
        if verbose > 1: #Debug
            print(total_flows.a)
            print(direction.a)
        alpha = bisect(lambda a : get_z_prime(a, total_flows, direction), 0, 1, disp=True)

        # Update
        flows_by_o.set_2d_array(flows_by_o.get_2d_array() + alpha*(direction_by_o.get_2d_array() - flows_by_o.get_2d_array()))
        total_flows.a = flows_by_o.get_2d_array().sum(axis=0)
    
    return flows_by_o, total_flows

def frankwolf_by_OD(net: Network, OD : np.array, shortest_path_alg = shortest_path, cost_function = BTR_cost_function, n_max=1e5, tolerance=1e-3, verbose=0):
    # Work in progress
    def direction_search(times : gt.EdgePropertyMap):
        """
            Performs all-or-nothing assignment based on the 'times' edge property
            Returns :
            - computed_flows : an EdgePropertyMap, which countains for each edge a vector with the flow from each origin
        """
        computed_flows = net.new_edge_property("vector<float>", vals=np.zeros((net.num_edges(), net.num_vertices())))
        flows_array = computed_flows.get_2d_array()

        # For each origin
        for o in range(OD.shape[0]):
            # Use shortest path algorithm to get list of links for each destination
            paths,_ = shortest_path_alg(net, times, o)
            # Convert list of links to numpy mask
            paths_mask = {k : net.get_edge_mask(i) for k,i in paths.items()}
            # Add flow of each destination to each link in its path
            for d, edge_mask in paths_mask.items():
                flows_array[o, edge_mask] += OD[o, d]
        
        # Set the array to the edge property
        computed_flows.set_2d_array(flows_array)

        return computed_flows
    
    def get_z_prime(alpha: float, flows: gt.EdgePropertyMap, direction: gt.EdgePropertyMap):
        times = BTR_cost_function(flows.a + alpha*(direction.a - flows.a), net)
        return ((direction.a - flows.a) * times.a).sum()

    # Initialisation
    flows_by_o = net.new_edge_property("vector<float>", vals=np.zeros((net.num_edges(), net.num_vertices())))
    total_flows = net.new_edge_property("float", vals=flows_by_o.get_2d_array().sum(axis=0))

    times = BTR_cost_function(total_flows.a, net)
    prev_total_time = 1e10 
    total_time = times.a.sum()

    flows_by_o = direction_search(times)
    total_flows.a = flows_by_o.get_2d_array().sum(axis=0)

    #Loop
    n_iter = 0
    def generator():
        while abs(total_time - prev_total_time) > tolerance and n_iter < n_max:
            yield
    if verbose >0: #For following the progress
        generator = tqdm(generator())
    else :
        generator = generator()
    for _ in generator:
        n_iter = n_iter + 1


        # Update time
        times = BTR_cost_function(total_flows.a, net)
        prev_total_time = total_time
        total_time = times.a.sum()

        # Direction search
        direction_by_o = direction_search(times)
        direction = net.new_edge_property("float", vals=direction_by_o.get_2d_array().sum(axis=0))

        # Line search
        if verbose > 1: #Debug
            print(total_flows.a)
            print(direction.a)
        alpha = bisect(lambda a : get_z_prime(a, total_flows, direction), 0, 1, disp=True)

        # Update
        flows_by_o.set_2d_array(flows_by_o.get_2d_array() + alpha*(direction_by_o.get_2d_array() - flows_by_o.get_2d_array()))
        total_flows.a = flows_by_o.get_2d_array().sum(axis=0)
    
    return flows_by_o, total_flows