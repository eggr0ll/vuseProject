'''
This file turns output from getcfg.py script into graphs.
'''

import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import os
import random
import dill
import re
from collections import defaultdict
from torch_geometric.data import Data
from torch.nn import Embedding
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader



# STEP 1: TOKENIZE ASSEMBLY INSTRS
# Breaks up assembly code into words
def tokenize_instructions(instruction):
    return instruction.split()

# initialize vocab dict that will contain words & their indices
vocab = defaultdict(lambda: len(vocab))


dir_list = os.listdir("./results_jp/01-runGenCfg/")
#print(dir_list)


all_graphs_max_length = 0;  # max number of features across all graphs
class_labels = []           # holds all malware types
graphs_as_data = []         # holds all 'data' objects
digraphs = []                 # holds all 'DiGraph' objects


# for each graph output file in results directory
for file in dir_list:
    G = nx.DiGraph() 
    
    full_file_path = "./results_jp/01-runGenCfg/" + file
    print("Processing", full_file_path)
    tokenized_instrs = []
    blocks = {}
    with open(full_file_path) as o_f:
        for count, line in enumerate(o_f):
            pass
    total_lines = count + 1

    with open(full_file_path) as output_file:
        # create blocks & successors dicts
        instr_in_block = []
        successors = {}
        for x in range(total_lines):
            line = output_file.readline()
            if "BLOCK" in line:
                line = line.split()
                block_address = int(line[1])
                block_successors = line[4].replace("[", "").replace("]", "").replace(",", " ").strip().split()
                block_successors = [eval(successor) for successor in block_successors] # cast to int
            elif "BLOCK" not in line and "None" not in line and "---" not in line:
                #instr_in_block.append(line[8:].strip().replace(",", ""))
                line = line.strip()
                pattern = re.compile(r'^\S+\s+(.*)')
                match = pattern.match(line)
                if match:
                    extracted_line = match.group(1).replace(",", "")
                    instr_in_block.append(extracted_line)
            # need to loop thru all instr b4 assigning it to key
            elif "None" in line:
                blocks[block_address] = instr_in_block
                instr_in_block = []
                successors[block_address] = block_successors
                block_successors = []
    #print(blocks)

    # tokenize instructions & add to vocab dict
    for instrs in blocks.values():
        # tokenize all instructions in block
        tok_block = [tokenize_instructions(instr) for instr in instrs]
        tokenized_instrs.append(tok_block)
        #print("Tokenized block:\n", tok_block)

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
        #print("Token indices:", token_indices)
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

    #print("Nodes:\n", G.nodes(data=True))
    #print("Edges:\n", G.edges())

    # STEP 4: CONVERT TO INPUT FOR PYGEO
    # need x = node feature matrix w/ [num_nodes, num_node_features] as tensor
    # need edge_index = graph connectivity in COO format w/ [2, num_edges] as tensor
    all_node_features = []      # stores embedding vector for each node in graph
    max_length = 0

    for node in G.nodes(data=True):
        if 'instructions' in node[1]:
            flattened_node_instrs = [item for sublist in node[1]['instructions'] for item in sublist]
            #print("Length of flattened node instrs:", len(flattened_node_instrs))
            max_length = max(max_length, len(flattened_node_instrs))
    print("Max length:", max_length)
    # if num of features within this graph is the max among all graphs
    if max_length > all_graphs_max_length:
        all_graphs_max_length = max_length

    # data=True means node is tuple where node[0] is address and node[1] is dictionary of attributes
    for node in G.nodes(data=True):      
        # iterates over each element in list of embedded vectors & flattens into 1 embedded vector list
        if 'instructions' in node[1]:
            flattened_node_instrs = [item for sublist in node[1]['instructions'] for item in sublist]
            #print("Flattened node instrs:", flattened_node_instrs)
            flattened_node_instrs = np.pad(flattened_node_instrs, (0, max_length - len(flattened_node_instrs)))     # pad w/ zeroes if list is too short
            #print("Flattened node instrs padded:", flattened_node_instrs)
            all_node_features.append(flattened_node_instrs)
            #print(f"Node {node[0]} padded length is {len(flattened_node_instrs)}")
        else:
            print(f"Node {node[0]} does not have 'instructions' attribute")

    digraphs.append(G)
    
    #print("All node features:", all_node_features)

    # convert node features to tensor of type float
    all_node_features_np = np.array(all_node_features)
    print("All node features padded:", all_node_features_np)
    x = torch.tensor(all_node_features_np, dtype=torch.float)
    print("Node feature matrix shape:", {x.shape})

    # create edge index tensor
    # converts edges into list of tuples -> converts list into tensor -> transposes tensor to have required shape [2, num_edges]
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    # create graph label
    graph_label = []                            # will hold graph label/type of malware as int
    underscore_idx = file.find("_")
    malware_type = file[:underscore_idx]        # extracts malware type string from front of filename
    if malware_type not in class_labels:
        class_labels.append(malware_type)       # adds malware type string to all class labels 
    print("Class labels:", class_labels)

    index = class_labels.index(malware_type)
    print("Index:", index)
    graph_label.append(index)                   # gets index of malware type 
    y = torch.tensor(graph_label)               # converts index of malware type into tensor
    print("Graph label:", y)

    # map OG node labels to contiguous "indices"
    unique_labels = torch.unique(edge_index)    # gets node numbers
    # creates dictionary mapping OG node labels to indices
    label_to_index = {label.item(): index for index, label in enumerate(unique_labels)}    
    new_edge_index = torch.tensor([[label_to_index[label.item()] for label in edge_pair] for edge_pair in edge_index.t()], dtype=torch.long).t().contiguous()
    print("Edited edge index tensor:", new_edge_index)

    # create obj describing graph
    data = Data(x=x, edge_index=new_edge_index, y=y)
    graphs_as_data.append(data)

    print("Node feature matrix:", x)
    #print("Edge index tensor:", edge_index)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of node features:", len(x[1]))
    #num_node_features = len(x[1])
    print("Done processing", full_file_path)
    print()

print("\nMax feature length of all graphs:", all_graphs_max_length)


# pad all graphs across all files to same length
print("Begin padding all graphs:\n")
for data, file in zip(graphs_as_data, dir_list):
    print("Processing", file)

    data_x_padded = F.pad(data.x, pad=(0, all_graphs_max_length - data.x.shape[1], 0, 0))
    data.x = data_x_padded
    print("Node feature matrix:", data.x)
    print("Node feature matrix shape:", data.x.shape)
    
    print("Done processing", file)
    print()
print("Done padding all graphs.")

with open("graphsAsData.pkl", "wb") as file:
    dill.dump(graphs_as_data, file)
with open("allGraphsMaxLength.pkl", "wb") as file:
    dill.dump(all_graphs_max_length, file)
