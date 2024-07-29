import networkx
import angr
import pprint
import argparse
#import logging
#logging.getLogger('angr').setLevel('DEBUG')

parser = argparse.ArgumentParser()
parser.add_argument("--input")
args = parser.parse_args()

p = angr.Project(args.input, load_options={'auto_load_libs': False})
#p = angr.Project('C:\\Users\\mango1\\Desktop\\sourcecode_datasets\\datasets\\Spyware\\Zeus\\bin\\7z.exe', load_options={'auto_load_libs': False})


# get names of functions in project
function_names = []
for f in p.kb.functions.items():
    print(f[1].name)
    function_names.append(f[1].name)
#    print(f[1].name)


# conduct CFG analysis
cfg = p.analyses.CFGFast()


# get graph for function main


for function_item in cfg.kb.functions.items():
    f = cfg.kb.functions[function_item[0]].graph

    for i in f.adjacency():
        # edited to prevent 'NoneType' object has no attribute 'block' error
        node = cfg.get_any_node(i[0].addr)
        the_block = None
        if node:
            the_block = node.block

        print('BLOCK {} has SUCCESSORS [{}] '.
            format(
                i[0].addr,
                ','.join([str(x.addr) for x in i[1].keys()])
                )
    
            )
        if the_block:
            print(the_block.pp())
        print('---')