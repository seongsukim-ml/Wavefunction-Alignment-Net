from torch_cluster import radius_graph
import torch
from torch_geometric.data import Data

def get_full_graph(batch_data):
    full_edge_index = []
    # radius_graph(batch_data.pos, 1000, batch_data.batch,max_num_neighbors=1000)
    # batch_data["non_diag_hamiltonian"] = batch_data["non_diag_hamiltonian"][full_edge_index[0]>full_edge_index[1]]
    # batch_data['non_diag_mask'] = batch_data["non_diag_mask"][full_edge_index[0]>full_edge_index[1]]
    # full_edge_index = full_edge_index[:,full_edge_index[0]>full_edge_index[1]]
    atom_start = 0
    for n_atom in batch_data["molecule_size"].reshape(-1):
        n_atom = n_atom.item()
        full_graph = torch.stack([torch.arange(n_atom).reshape(-1,1).repeat(1,n_atom),torch.arange(n_atom).reshape(1,-1).repeat(n_atom,1)],axis = 0).reshape(2,-1)
        full_graph = full_graph[:,full_graph[0]!=full_graph[1]]
        full_edge_index.append(atom_start+full_graph)
        atom_start = atom_start + n_atom

    return torch.concat(full_edge_index,dim = 1).to(batch_data["molecule_size"].device)

def get_transpose_index(data, full_edges):
    start_edge_index = 0
    all_transpose_index = []
    for graph_idx in range(data.ptr.shape[0] - 1):
        num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
        graph_edge_index = full_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
        sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
        bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
        transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
        transpose_index = transpose_index + start_edge_index
        all_transpose_index.append(transpose_index)
        start_edge_index = start_edge_index + num_nodes*(num_nodes-1)
    return torch.cat(all_transpose_index, dim=-1)

def get_transpose_index(data, full_edges):
    start_edge_index = 0
    all_transpose_index = []
    for graph_idx in range(data.ptr.shape[0] - 1):
        num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
        graph_edge_index = full_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
        sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
        bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
        transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
        transpose_index = transpose_index + start_edge_index
        all_transpose_index.append(transpose_index)
        start_edge_index = start_edge_index + num_nodes*(num_nodes-1)
    return torch.cat(all_transpose_index, dim=-1)

def get_toy_data(n_atom = 100):
    out = {}
    neighbors = 30
    atom_max_orbital = 14  # def2-svp 14 def2-tzvp 37
    pos = torch.randn(n_atom,3)
    atomic_numbers = torch.randint(low = 0,high=20,size = (n_atom,))

    edge_index = radius_graph(pos, 10,max_num_neighbors=neighbors)
    energy = torch.randn(1)
    forces = torch.randn(n_atom,3)
    diag_hamiltonian = torch.randn((n_atom,atom_max_orbital,atom_max_orbital))
    non_diag_hamiltonian = torch.randn((n_atom*(n_atom-1),atom_max_orbital,atom_max_orbital))
    diag_mask = torch.ones((n_atom,atom_max_orbital,atom_max_orbital))
    non_diag_mask = torch.ones((n_atom*(n_atom-1),atom_max_orbital,atom_max_orbital))

    out.update({
        "molecule_size":torch.Tensor([n_atom]).int(),
        "pos":pos,
        "atomic_numbers":atomic_numbers,
        "edge_index":edge_index,
        "energy":energy,
        "forces":forces,
        "diag_hamiltonian":diag_hamiltonian,
        "non_diag_hamiltonian":non_diag_hamiltonian,
        "diag_mask":diag_mask,
        "non_diag_mask":non_diag_mask
    })
    return out
    # Data(edge_index=[2, 32], pos=[9, 3], interaction_graph=[2, 9], labels=[9], 
    #      num_nodes=9, num_labels=2, label_batch=[2], diag_hamiltonian=[9, 14, 14], 
    #      mask_l1=[32], non_diag_mask=[32, 14, 14], batch=[9], energy=[2], 
    #      diag_mask=[9, 14, 14], atomic_numbers=[9], molecule_size=[2], forces=[9, 3], 
    #      non_diag_hamiltonian=[32, 14, 14])
def construct_o3irrps(dim,order):
    string = []
    for l in range(order+1):
        string.append(f"{dim}x{l}e" if l%2==0 else f"{dim}x{l}o")
    return "+".join(string)

def to_torchgeometric_Data(data:dict):
    torchgeometric_data = Data()
    for key in data.keys():
        torchgeometric_data[key] = data[key]
    return torchgeometric_data

def construct_o3irrps_base(dim,order):
    string = []
    for l in range(order+1):
        string.append(f"{dim}x{l}e")
    return "+".join(string)