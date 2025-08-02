import sys
# from cmath import inf
from distutils.log import info
from functools import cmp_to_key
# from tokenize import Double
from pyscf import gto, scf
import numpy as np
import networkx as nx
from itertools import combinations
# from lightnp.data.datasets.Hut_conformer import parseRDKitMol
# from tool.cpu_dft_hook import NumIntHook
# from tool.CuDFT.src.pyscf_binding.cuda_factory import CUDAFactory
from pyscf.data.elements import _atom_symbol
import math

atomic_radii = dict(Ac=1.88, Ag=1.59, Al=1.35, Am=1.51, As=1.21, Au=1.50, B=0.83, Ba=1.34, Be=0.35, Bi=1.54, Br=1.21,
                    C=0.68, Ca=0.99, Cd=1.69, Ce=1.83, Cl=0.99, Co=1.33, Cr=1.35, Cs=1.67, Cu=1.52, D=0.23, Dy=1.75,
                    Er=1.73, Eu=1.99, F=0.64, Fe=1.34, Ga=1.22, Gd=1.79, Ge=1.17, H=0.23, Hf=1.57, Hg=1.70, Ho=1.74,
                    I=1.40, In=1.63, Ir=1.32, K=1.33, La=1.87, Li=0.68, Lu=1.72, Mg=1.10, Mn=1.35, Mo=1.47, N=0.68,
                    Na=0.97, Nb=1.48, Nd=1.81, Ni=1.50, Np=1.55, O=0.68, Os=1.37, P=1.05, Pa=1.61, Pb=1.54, Pd=1.50,
                    Pm=1.80, Po=1.68, Pr=1.82, Pt=1.50, Pu=1.53, Ra=1.90, Rb=1.47, Re=1.35, Rh=1.45, Ru=1.40, S=1.02,
                    Sb=1.46, Sc=1.44, Se=1.22, Si=1.20, Sm=1.80, Sn=1.46, Sr=1.12, Ta=1.43, Tb=1.76, Tc=1.35, Te=1.47,
                    Th=1.79, Ti=1.47, Tl=1.55, Tm=1.72, U=1.58, V=1.33, W=1.37, Y=1.78, Yb=1.94, Zn=1.45, Zr=1.56)


class MolGraph:
    """Represents a molecular graph."""
    __slots__ = ['elements', 'x', 'y', 'z', 'adj_list',
                 'atomic_radii', 'bond_lengths']

    def __init__(self, atom_types, atom_cords):
        self.elements = []
        self.x = []
        self.y = []
        self.z = []
        self.adj_list = {}
        self.atomic_radii = []
        self.bond_lengths = {}
        self._build_graph(atom_types, atom_cords)

    # def read_xyz(self, file_path: str) -> None:
    #     """Reads an XYZ file, searches for elements and their cartesian coordinates
    #     and adds them to corresponding arrays."""
    #     pattern = re.compile(r'([A-Za-z]{1,3})\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)')
    #     with open(file_path) as file:
    #         for element, x, y, z in pattern.findall(file.read()):
    #             self.elements.append(element)
    #             self.x.append(float(x))
    #             self.y.append(float(y))
    #             self.z.append(float(z))
    #     self.atomic_radii = [atomic_radii[element] for element in self.elements]
    #     self._generate_adjacency_list()

    def _build_graph(self, atom_types, atom_cords):
        """Reads an XYZ file, searches for elements and their cartesian coordinates
        and adds them to corresponding arrays."""
        for i in range(len(atom_types)):
            self.elements.append(_atom_symbol(atom_types[i]))
            self.x.append(atom_cords[i][0])
            self.y.append(atom_cords[i][1])
            self.z.append(atom_cords[i][2])
        self.atomic_radii = [atomic_radii[element] for element in self.elements]
        self._generate_adjacency_list()

    def _generate_adjacency_list(self):
        """Generates an adjacency list from atomic cartesian coordinates."""
        node_ids = range(len(self.elements))
        for i, j in combinations(node_ids, 2):
            x_i, y_i, z_i = self.__getitem__(i)[1]
            x_j, y_j, z_j = self.__getitem__(j)[1]
            distance = math.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
            if 0.1 < distance < (self.atomic_radii[i] + self.atomic_radii[j]) * 1.3:
                self.adj_list.setdefault(i, set()).add(j)
                self.adj_list.setdefault(j, set()).add(i)
                self.bond_lengths[frozenset([i, j])] = round(distance, 5)

    # with for recerence, too slow.
    def build_adj_matrix(self,):
        _adj = np.zeros((len(self.elements),len(self.elements)))
        node_ids = range(len(self.elements))
        for i, j in combinations(node_ids, 2):
            x_i, y_i, z_i = self.__getitem__(i)[1]
            x_j, y_j, z_j = self.__getitem__(j)[1]
            distance = math.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
            if 0.1 < distance < (self.atomic_radii[i] + self.atomic_radii[j]) * 1.3:
                _adj[i, j] = round(distance, 5)
                _adj[j, i] = round(distance, 5)
        return _adj

    def edges(self):
        """Creates an iterator with all graph edges."""
        edges = set()
        for node, neighbours in self.adj_list.items():
            for neighbour in neighbours:
                edge = frozenset([node, neighbour])
                if edge in edges:
                    continue
                edges.add(edge)
                yield node, neighbour

    def to_networkx_graph(self):
        """Creates a NetworkX graph.
        Atomic elements and coordinates are added to the graph as node attributes 'element' and 'xyz" respectively.
        Bond lengths are added to the graph as edge attribute 'length''"""
        G = nx.Graph(self.adj_list)
        node_attrs = {num: {'element': element, 'xyz': xyz} for num, (element, xyz) in enumerate(self)}
        nx.set_node_attributes(G, node_attrs)
        edge_attrs = {edge: {'length': length} for edge, length in self.bond_lengths.items()}
        nx.set_edge_attributes(G, edge_attrs)
        return G

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, position):
        return self.elements[position], (
            self.x[position], self.y[position], self.z[position])

def process_mol(xyz_file):
    atom_types = []
    atom_cords = []
    with open(xyz_file, 'r') as f:
        for line in f:
            line = line.split()
            atom_types.append(line[0])
            atom_cords.append([float(i) for i in line[1:]])
    atom_cords = np.array(atom_cords, dtype=np.float64)
    atom_dist_matrix =np.sqrt(np.sum((atom_cords.reshape(-1, 1, 3) - atom_cords.reshape(1, -1, 3))**2, axis=-1))

    return atom_types, atom_cords, atom_dist_matrix


def find_edge_sup(atom_dist_matrix, groups, num_limit=2):
    '''For each atom, find the top `num_limit` nearest ignored edges'''
    natom = len(atom_dist_matrix)
    atom2group = [0]*natom

    #Build mapping from atom to groups
    for idx, g in enumerate(groups):
        for a in g:
            atom2group[a] = idx

    edge_groups = []
    sort_idx = np.argsort(atom_dist_matrix, axis=-1)
    for i in range(natom):
        for j in sort_idx[i,1:num_limit+1]:
            if j>i and atom2group[i] != atom2group[j]: # i and j are not in the same groups
                edge_groups.append([i,j])
    return edge_groups

def count_overlap(natom, groups):
    '''Count how many times each atom is used'''
    ol_cnt = [0] * natom
    for g in groups:
        for a in g:
            ol_cnt[a] += 1
    return ol_cnt

def build_graph_with_distance(atom_dist_matrix, distance_cutoff=2.0):
    G = nx.Graph()
    natom = len(atom_dist_matrix)
    sorted_idx = np.argsort(atom_dist_matrix)
    for i in range(natom):
        for j in sorted_idx[i, 1:]:
            if atom_dist_matrix[i, j] <= distance_cutoff:
                G.add_edge(i, j)
            else:
                break
    assert(nx.is_connected(G))
    return G

def build_graph_from_xyz(xyz_filename):
    mg = MolGraph()
    mg.read_xyz(xyz_filename)
    G = mg.to_networkx_graph(mg)
    return G

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                #    is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                #    bond_type=bond.GetBondType(),
                )
        
    return G

def build_graph(atom_types, atom_cords, method="molecule", distance_cutoff=2.0):
    """
    Building graph from atom types and atom coordinates.

    Args:
        atom_types (list): list of atom types.
        atom_cords (list): list of atom coordinates.
        method (str): `molecule` or `distance`. For the molecule method, the graph will be built 
                      according to real chemical bond, while for the distance method, the graph 
                      will be build according to the `distance_cutoff`.
        distance_cutoff (float): the distance cutoff of edges for the `distance` method. 
    """
    G = nx.Graph()
    if method == "molecule":
        mg = MolGraph(atom_types, atom_cords)
        G = mg.to_networkx_graph()
    elif method == "distance":
        atom_types, atom_cords = np.array(atom_types), np.array(atom_cords)
        atom_dist_matrix =np.sqrt(np.sum((atom_cords.reshape(-1, 1, 3) - atom_cords.reshape(1, -1, 3))**2, axis=-1))
        G = build_graph_with_distance(atom_dist_matrix, distance_cutoff)
    else:
        raise NotImplementedError

    return G

def BFS(G, src, num_limit=None, mask={}):
    '''Conduct BFS on `G` from `src` node. Up to `num_limit` nodes are visited.'''
    if num_limit is None or num_limit <= 0:
        num_limit = len(G.nodes)
    # flags = {idx: 1 for idx in range(len(mask)) if mask[idx]}
    # flags[src] = 1
    if mask is None:
        mask = {}

    ls = [src]
    mask[src] = 1
    idx = 0
    while len(ls) < num_limit and idx < len(ls):
        for n in G.neighbors(ls[idx]):
            if mask.get(n, 0):
                continue
            ls.append(n)
            mask[n] = 1
            if len(ls) >= num_limit:
                break
        idx += 1
    
    return ls

# def sample_traj(traj_list, p=None):
#     n = len(traj_list)
#     if p is None:
#         # TODO: Find some hyper-parameters with more theoretical guarantee.
#         p = np.power([1.05]*n, range(n))
#         p /= np.sum(p)

#     idx = np.random.choice(n, p=p)
#     return idx

# def sample_node(traj, max, p=None):
#     n = min(len(traj), max)
#     if p is None:
#         # TODO: Find some hyper-parameters with more theoretical guarantee.
#         p = np.power([1.05]*n, range(n))
#         p /= np.sum(p)

#     idx = np.random.choice(n, p=p)
#     return idx

def BFS_with_mask(G, mask):
    visited = dict(mask)
    traj_list = []
    rand_list = list(G.nodes)
    np.random.shuffle(rand_list)
    for node in rand_list:
        # print(len(G.nodes()), node, len(visited))
        if visited.get(node, 0):
            continue
        traj = BFS(G, node, num_limit=len(G.nodes), mask=visited)
        traj_list.append(traj)

    return traj_list

def generate_mask(n, visit_cnt, limits):
    '''
    At most `limits` nodes among `n` nodes are masked with probability proportional to `visit_cnt`.
    '''
    mask_cnt = np.random.randint(limits)
    cnt = np.array(visit_cnt)
    cnt[cnt>300] = 300
    p = np.power(10.0, cnt)
    p /= p.sum()
    sample = np.random.choice(n, mask_cnt, p = p)
    mask = {idx:0 for idx in range(n)}
    for node in sample:
        mask[node] = 1

    return mask

def find_possible_subgraph(G, subgraph_size, graph_num_limits=100):
    '''
    Step 1. Random generate a mask for the BFS trajectory. \n
    Step 2. Perform BFS with the mask to find all trajectories greater than the `subgraph_size`. \n
    Step 3. Truncate the trajectory to construct a subgraph with specified size and insert in to `subgraph_list` if haven't seen before. \n
    '''
    # assert(nx.is_connected(G))
    node_list = list(G.nodes)
    if len(node_list) <= subgraph_size:
        return [node_list]

    subgraph_list = []
    try_cnt = 0
    mask = {idx: 0 for idx in range(len(G.nodes))}
    visited_cnt = [0] * len(G.nodes)
    while len(subgraph_list) < graph_num_limits and try_cnt < graph_num_limits * 3:
        try_cnt += 1
        traj_list = BFS_with_mask(G, mask)
        for traj in traj_list:
            if len(traj) < subgraph_size:
                continue
            traj = sorted(traj[:subgraph_size])
            if traj not in subgraph_list:
                subgraph_list.append(traj)
                for node in traj:
                    visited_cnt[node] += 1

        mask = generate_mask(len(G.nodes), visited_cnt, (len(G.nodes)-subgraph_size)//2)

    return subgraph_list

# def find_possible_subgraph(G, subgraph_size, graph_num_limits=1000):
#     '''
#     Step 1. Construct a BFS trajectory from arbitrary node. \n
#     Step 2. Flip a random bit to remove a node `n` from the trajectory. \n
#     Step 3. Complete BFS without node `n`. \n
#     Remark: The random bit should be carefully designed to obtain as diverse subgraph as possible. \n
#     '''
#     assert(nx.is_connected(G))
#     node_list = list(G.nodes)
#     if len(node_list) <= subgraph_size:
#         return [node_list]

#     # Step 1
#     bfs_traj = BFS(G, len(node_list))
#     traj_list = [bfs_traj]
#     mask_list = [[0]*len(bfs_traj)]
#     subgraph_list = [bfs_traj[:subgraph_size]]
#     try_cnt = 0
#     while len(subgraph_list) < graph_num_limits and try_cnt < graph_num_limits * 1.5:
#         try_cnt += 1

#         # Step 2
#         traj_idx = sample_traj(traj_list)
#         node_idx = sample_node(traj_list[traj_idx], max=subgraph_size)

#         # Step 3
#         new_mask = list(mask_list[traj_idx])
#         new_mask[node_idx] = 1
#         new_traj = BFS_with_mask(G, mask_list[traj_idx])
#         if len(new_traj) < subgraph_size:
#             continue
#         # Note that we haven't handle repeated traj here.
#         traj_list.append(new_traj)
#         mask_list.append(new_mask)
#         subgraph_list.append(new_traj[:subgraph_size])

#     return subgraph_list

def find_possible_fragmentation(G, num_limit, frag_count):
    pass

def compare_groups(g1, g2):
    l1, l2 = len(g1), len(g2)
    if l1 != l2:
        return l1 - l2
    if g1 == g2:
        return 0
    if g1 < g2:
        return -1
    else:
        return 1

def find_min_group_id(G, node_list, groups, atom2group):
    '''Find the smallest group associated with the points in the `node_list`'''
    gidx = -1
    gsize = 1e10
    for n in node_list:
        for nb in list(G[n]):
            if nb in node_list:
                continue
            tmp_idx = atom2group[nb]
            tmp_size = len(groups[tmp_idx])
            if tmp_size < gsize:
                gidx = tmp_idx
                gsize = tmp_size

    return gidx
        
def balanced_graph_grouping(G, num_limit=6, shuffle=True):
    '''
    Build groups according to `G`, with the sizes of groups as close to the `num_limit` as possible. \n
    When `shuffle` is `True`, a random fragmentation starting point.
    '''
    GC = G.copy()
    natom = G.number_of_nodes()
    groups = []
    atom2group = [-1]*natom
    iso_nodes = []
    while GC.number_of_nodes() > 0:
        min_degree = natom + 1
        min_node = -1

        # Search for isolated nodes and non-isolated node with minimum degree
        all_nodes = list(GC.nodes)
        if shuffle:
            np.random.shuffle(all_nodes)
        for n in all_nodes:
            if GC.degree[n] == 0: # Find an isolate node
                iso_nodes.append(n)
            elif GC.degree[n] < min_degree:
                min_degree = GC.degree[n]
                min_node = n
        # Isolated nodes are removed from the diagram and processed later
        GC.remove_nodes_from(iso_nodes)

        # The node with the lowest degree was not found
        if min_degree == natom + 1:
            break

        # Starting from the node with the smallest degree, groups are established through breadth-first search
        ls = BFS(GC, min_node, num_limit)
        for i in ls:
            atom2group[i] = len(groups)
        groups.append(ls)
        GC.remove_nodes_from(ls)

    # Uniformly handle isolated nodes
    for n in iso_nodes:
        gidx = find_min_group_id(G, [n], groups, atom2group)
        if gidx == -1:
            continue
        groups[gidx].append(n)
        atom2group[n] = gidx

    # For smaller groups, they will be merged into adjacent groups as much as possible. 
    group_len = [len(g) for g in groups]
    group_sort_idx = np.argsort(group_len)
    for idx in group_sort_idx:
        g = groups[idx]
        if len(g) > num_limit*0.6:
            continue
        gidx = find_min_group_id(G, g, groups, atom2group)
        if gidx == -1 or len(groups[gidx]) > num_limit + 2: # The smallest nearest group is already too large to be merged
            continue
        groups[gidx].extend(g)
        for n in g:
            atom2group[n] = gidx
        groups[idx] = []

    # Remove invalid groups
    groups = [g for g in groups if g]
    return groups

def cord2xyz(atom_types, atom_cords):
    xyz = ""
    for i in range(len(atom_cords)):
        xyz += f"{atom_types[i]} {' '.join([str(j) for j in atom_cords[i]])}\n"
    return xyz

# def cal_dft(atom_types, atom_cords, cal_mode='cpu', max_mesh_len = 12): 
#     xyz = cord2xyz(atom_types, atom_cords)
#     info = {'energy':np.array([0]), 'rho':np.array([[0]])}
#     if xyz == None:
#         return info
#     mol = gto.M(atom=xyz, basis='6-31G(d)', cart=True, spin=None)
#     if mol.nelectron % 2 == 0:
#         ks = scf.RKS(mol)
#     else:
#         print("oops!!")
#         ks = scf.UKS(mol)
#     # mol.verbose=9
#     ks.xc = "M06-2X"
#     ks.grids.level=2
#     # ks.init_guess="vasp"
#     mol.max_memory=128000
#     ks.max_memory=128000
#     ks.max_cycle=30
#     ks.direct_scf_tol=1e-10 # previous is 1e-12
#     ks.conv_tol=1e-6
#     if cal_mode == 'cpu':
#         ks._numint = NumIntHook()
#         ks.kernel()
#         info['rho'] = ks._numint.get_den_mesh(mol, ks.xc, ks.max_memory, atom_cords, max_mesh_len, resolution= 20)
#         info['dms'] = ks._numint.dms.reshape(-1)
#     elif cal_mode == 'gpu':
#         ks = CUDAFactory.generate_cuda_instance(ks)
#         ks.kernel()
#         info['rho'] = ks._numint.get_den_mesh(mol, ks.xc, ks.max_memory, atom_cords, max_mesh_len, resolution= 20)
#         info['dms'] = ks._numint.integrator.dms.reshape(-1)
    
#     info['energy'] = [ks.energy_tot()]
#     # dense_rho = ks._numint.rho
#     # dense_coords = ks._numint.coords
#     #info['rho'] = ks._numint.sample_den_mesh(mol, resolution=20) # downsampling
#     info['atom_types'] = atom_types.reshape(-1)
#     info['atom_cords'] = atom_cords.reshape(-1)

#     return info

# def cal_tot_group_energy(pieces, groups, cal_mode, addH):
#     tot_info = {}
#     clean_groups = []
#     clean_groups_size = np.array([])
#     if len(groups) == 0:
#         return cal_dft(None, None)
#     for i, piece in enumerate(pieces):
#         group, atom_types, atom_coords = parseRDKitMol(piece, addH, groups[i])
#         clean_groups.append((group))
#         clean_groups_size=np.append(clean_groups_size, len(group))
#         info = cal_dft(atom_types, atom_coords, cal_mode)   

#         for item in info:
#             new_info = np.expand_dims(np.array(info[item]), axis=0)
#             if item in tot_info.keys():
#                 pad_len = new_info.shape[-1] - tot_info[item].shape[-1]
#                 if pad_len < 0:
#                     new_info = np.pad(new_info,((0,0),(0,abs(pad_len))),'constant', constant_values=(0,0))
#                 elif pad_len > 0:
#                     tot_info[item] = np.pad(tot_info[item],((0,0),(0,abs(pad_len))),'constant', constant_values=(0,0))
#                 tot_info[item] = np.concatenate((tot_info[item], new_info), axis=0)  
#             else:
#                 tot_info[item] = new_info
#         # tot += energy
#     tot_info['clean_size'] = clean_groups_size
#     return clean_groups, tot_info #tot, energy_list, rho_list

def sort_groups(groups):
    for g in groups:
        g.sort()
    groups.sort(key=cmp_to_key(compare_groups))
    return groups

if __name__ == '__main__':
    atom_types, atom_cords, atom_dist_matrix = process_mol("85-porphy.xyz")
    G = build_graph(atom_types, atom_cords, method="molecule")
    # traj_list = find_possible_subgraph(G, 50, 1000)
    # for traj in traj_list:
    #     sg = G.subgraph(traj)
    #     if not nx.is_connected(sg):
    #         print(traj)
    # print(len(traj_list))
    # sys.exit(0)
    
    #G = build_graph_with_distance(atom_dist_matrix, 2)

    # Calculate the energy of each grouping
    groups = balanced_graph_grouping(G, 6, mode="shuffle") 
    for g in groups:
        g.sort()

    groups.sort(key=cmp_to_key(compare_groups))
    for g in groups:
        print(g)
    sys.exit(0)
    tot = cal_tot_group_energy(atom_types, atom_cords, groups)
    print(f"tot: {tot}")

    # Calculate the energy of the edges broken by grouping
    edge_groups = find_edge_sup(atom_dist_matrix, groups, 2)
    print(f"Edges supp: {len(edge_groups)}")
    edge_tot = cal_tot_group_energy(atom_types, atom_cords, edge_groups)
    print(f"edge_tot: {edge_tot}")

    # Calculate the atomic energy repeated many times
    ol_cnt = count_overlap(len(atom_types), groups+edge_groups)
    ol_tot = 0
    atom_energy_dict = {}
    print(f"Overlap cnt: {sum(ol_cnt)}")
    print(ol_cnt)
    for idx, a in enumerate(atom_types):
        if a not in atom_energy_dict.keys():
            atom_energy_dict[a],_,_ = cal_dft(atom_types[idx], atom_cords[idx], 'cpu')
        
        ol_tot += max(0, (ol_cnt[idx]-1))*atom_energy_dict[a]

    final_energy = tot + edge_tot - ol_tot

    print(tot, edge_tot, ol_tot, final_energy)