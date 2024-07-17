'''
This file turns output from getcfg.py script into graphs.
'''

import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from torch_geometric.data import Data
from torch.nn import Embedding
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


G = nx.DiGraph()

# STEP 1: TOKENIZE ASSEMBLY INSTRS
# Breaks up assembly code into words
def tokenize_instructions(instruction):
    return instruction.split()

# initialize vocab dict that will contain words & their indices
vocab = defaultdict(lambda: len(vocab))

tokenized_instrs = []
blocks = {}
with open("results/Botnet_Android-GMBot-MALWARE-master_Android-GMBot-MALWARE-master_windows_gozi-isfb_appack.exe.output") as o_f:
    for count, line in enumerate(o_f):
        pass
total_lines = count + 1

with open("results/Botnet_Android-GMBot-MALWARE-master_Android-GMBot-MALWARE-master_windows_gozi-isfb_appack.exe.output") as output_file:
    # create blocks & successors dicts
    instr_in_block = []
    successors = {}
    for x in range(total_lines):
        line = output_file.readline()
        if "BLOCK" in line:
            block_address = int(line[6:13])
            block_successors = line[30:].replace(",", " ").replace("]", "").strip().split()
            block_successors = [eval(successor) for successor in block_successors] # cast to int
        elif "BLOCK" not in line and "None" not in line and "---" not in line:
            instr_in_block.append(line[8:].strip().replace(",", ""))
        # need to loop thru all instr b4 assigning it to key
        elif "None" in line:
            blocks[block_address] = instr_in_block
            instr_in_block = []
            successors[block_address] = block_successors
            block_successors = []
print(blocks)

# tokenize instructions & add to vocab dict
for instrs in blocks.values():
    # tokenize all instructions in block
    tok_block = [tokenize_instructions(instr) for instr in instrs]
    tokenized_instrs.append(tok_block)
    print("Tokenized block:\n", tok_block)

    # add each token to vocab
    for instr in tok_block:
        for token in instr:
            _ = vocab[token]

#print("Tokenized instructions:\n", tokenized_instrs)
#print("Vocab:\n", vocab)

#nx.draw(G)
#plt.show()
#print("Total Lines:", total_lines)

# STEP 2: PERFORM EMBEDDINGS
vocab_size = len(vocab)     # should be 23
embedding_dim = 10          # experiment w/ changing
embedding_layer = Embedding(vocab_size, embedding_dim)

# returns single 1D vector that represents entire instruction by averaging embedding vectors of individual tokens
def embed_instr(instr):
    tokens = instr.split()
    token_indices = [vocab[token] for token in tokens]
    embedded_tokens = embedding_layer(torch.tensor(token_indices))
    return embedded_tokens.mean(dim=0)


# STEP 3: ADD NODES
for block_address, instructions in blocks.items():
    encoded_instrs = [embed_instr(instr).detach().numpy() for instr in instructions]
    G.add_node(block_address, instructions=encoded_instrs)

# STEP 3.5: ADD EDGES
#print("Successors:\n", successors)
for block_address, successors in successors.items():
    for successor in successors:
        G.add_edge(block_address, successor)
    
print("Nodes:\n", G.nodes(data=True))
print("Edges:\n", G.edges())



# STEP 4: CONVERT TO INPUT FOR PYGEO
# need x = node feature matrix w/ [num_nodes, num_node_features] as tensor
# need edge_index = graph connectivity in COO format w/ [2, num_edges] as tensor
all_node_features = []      # stores embedding vector for each node in graph
max_length = 0

for node in G.nodes(data=True):
    if 'instructions' in node[1]:
        flattened_node_instrs = [item for sublist in node[1]['instructions'] for item in sublist]
        max_length = max(max_length, len(flattened_node_instrs))
print("Max length:", max_length)

# data=True means node is tuple where node[0] is address and node[1] is dictionary of attributes
for node in G.nodes(data=True):      
    # iterates over each element in list of embedded vectors & flattens into 1 embedded vector list
    if 'instructions' in node[1]:
        flattened_node_instrs = [item for sublist in node[1]['instructions'] for item in sublist]
        flattened_node_instrs = np.pad(flattened_node_instrs, (0, max_length - len(flattened_node_instrs)))     # pad w/ zeroes if list is too short
        all_node_features.append(flattened_node_instrs)
        print(f"Node {node[0]} padded length is {len(flattened_node_instrs)}")
    else:
        print(f"Node {node[0]} does not have 'instructions' attribute")
    

# convert node features to tensor of type float
all_node_features_np = np.array(all_node_features)
x = torch.tensor(all_node_features_np, dtype=torch.float)
print("Node feature matrix shape:", {x.shape})

# create edge index tensor
# converts edges into list of tuples -> converts list into tensor -> transposes tensor to have required shape [2, num_edges]
edge_index = torch.tensor(list(G.edges)).t().contiguous()

# create obj describing graph
data = Data(x=x, edge_index=edge_index)

print("Node feature matrix:", x)
print("Edge index tensor:", edge_index)



'''
# CREATE GNN
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(x[1].num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)       # 1 is sub for dataset.num_classes

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=64)
print(model)
'''