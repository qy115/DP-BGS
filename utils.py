

import os
import json

import argparse

from collections import defaultdict
from datetime import datetime

def save_graph(
        graph,
        output_dir,
        filename="synthetic_graph.txt",
        info_name = 'graph_info.json',
        save_info = False
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    edge_path=os.path.join(
        output_dir,
        filename
    )


    
    U_reverse=graph["U_reverse"]

    V_reverse=graph["V_reverse"]
    
    original_edges=[]
    
    for u,v in graph["edges"]:


        original_u=U_reverse[u]

        original_v=V_reverse[v]


        original_edges.append(
            (
                original_u,
                original_v
            )
        )
        


    original_edges.sort(
        key=lambda x:
        (
            int(x[0]),
            int(x[1])
        )
    )




    with open(
        edge_path,
        "w",
        encoding="utf-8"
    ) as f:


        for u,v in original_edges:


            f.write(
                f"{u},{v}\n"
            )

    print(
        "Synthetic graph saved:",
        edge_path
    )


    
    if save_info:
        info={
            "num_U":
                graph["num_U"],
            "num_V":
                graph["num_V"],
            "num_edges":
                len(graph["edges"]),
            "generated_time":
                str(datetime.now())
        }



        info_path=os.path.join(
            output_dir,
            info_name
        )



        with open(info_path, "w",encoding="utf-8") as f:
            json.dump(
                info,
                f,
                indent=4
            )


    

def save_results(
        results,
        output_dir,
        filename="evaluation.json"
):


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    path=os.path.join(
        output_dir,
        filename
    )



    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            results,
            f,
            indent=4
        )


    print("Evaluation results saved:",path)



def build_bipartite_graph(edges):


    U_original=set()
    V_original=set()



    for u,v in edges:
        U_original.add(u)
        V_original.add(v)



    U_original=list(U_original)
    V_original=list(V_original)



    U_mapping={
        old:i
        for i,old in enumerate(U_original)
    }


    V_mapping={
        old:i
        for i,old in enumerate(V_original)
    }



    U_reverse={
        i:old
        for old,i in U_mapping.items()
    }


    V_reverse={
        i:old
        for old,i in V_mapping.items()
    }



    internal_edges=[]


    for u,v in edges:


        u_id=U_mapping[u]
        v_id=V_mapping[v]

        internal_edges.append(
            (
                u_id,
                v_id
            )
        )





    adj_U=defaultdict(set)

    adj_V=defaultdict(set)



    for u,v in internal_edges:
        adj_U[u].add(v)
        adj_V[v].add(u)



    degree_U=[
        len(adj_U[i])
        for i in range(len(U_original))
    ]


    degree_V=[

        len(adj_V[i])
        for i in range(len(V_original))

    ]



    return {

        "U":
            list(range(len(U_original))),

        "V":
            list(range(len(V_original))),

        "edges":
            internal_edges,

        "adj_U":
            adj_U,

        "adj_V":
            adj_V,

        "degree_U":
            degree_U,


        "degree_V":
            degree_V,
            
        "num_U":
            len(U_original),

        "num_V":
            len(V_original),


        "U_mapping":
            U_mapping,


        "V_mapping":
            V_mapping,


        "U_reverse":
            U_reverse,


        "V_reverse":
            V_reverse

    }


def load_bipartite_graph(
        data_dir,
        data_name,
        sep
):

   

    file_path = os.path.join(
        data_dir,
        data_name
    )


    print(
        "Reading:",
        file_path
    )


    edges=[]


    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:


        for line in f:


            line=line.strip()


            if not line:
                continue


            parts=line.split(sep)


            if len(parts)<2:
                continue


            u=parts[0]
            v=parts[1]


            edges.append(( u, v ))



    graph = build_bipartite_graph(
        edges
    )

    return graph



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
