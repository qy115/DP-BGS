
import os, sys

current_directory = os.getcwd()
sys.path.append(current_directory)

import numpy as np
import argparse
import random
import time

from utils import save_graph, save_results, load_bipartite_graph, str2bool
from generate1_ab import *
from eval1 import evaluate_all


def parse_args():
    """Arguments"""
    parser = argparse.ArgumentParser(description="Obtain the performance of BKGS.")
    parser.add_argument('--path', nargs='?', default='./datasets/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pantry',
                        help='Choose a dataset.')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Privacy budget.')
    
    parser.add_argument('--seed_n', type=int, default=1,
                        help='Seed.')
    
    parser.add_argument('--ratio1', type=float, default=0.5,
                            help='The budget ratio of first part.')
    parser.add_argument('--ru', type=float, default=0.5,
                                help='The budget ratio of U-side nodes.')

    parser.add_argument('--lam', type=float, default=0.1,
                                help='The lambda parameter.')

    
    parser.add_argument('--redist', type=str2bool, default=True,
                        help='Whether to redistribute the degree.')

   

    parser.add_argument('--max_rounds', type=int, default=50,
                                help='Maximum rounds.')

    parser.add_argument('--mode', nargs='?', default='degree',
                            help='Choose a mode.')

    return parser.parse_args()





if __name__ == '__main__':

    
    args = parse_args()

    print("params: %s " % (args))

    seed = args.seed_n
    data_dir = args.path

    dataset_name = args.dataset
    dataset_path = dataset_name + '.txt'

    eps = args.epsilon
    eps_e = min(0.001, eps*0.01)


    ratio1 = args.ratio1
    ru = args.ru
   
    max_rounds = args.max_rounds

    lam = args.lam
    th_coeff = np.sqrt(2/lam)
    
    redist = args.redist

    sep = ','

    mode = args.mode

    
    np.random.seed(seed)
    random.seed(seed)

    output_path = f"./results/{dataset_name}/"
        
    if mode == 'judge':
        m1_name = f'JBGS_{dataset_name}_eps_{eps}_ratio1_{ratio1}_ru_{ru}_lam_{lam}_redist_{redist}_mr_{max_rounds}_seed_{seed}'
    elif mode == 'block':
        m1_name = f'BKGS_{dataset_name}_eps_{eps}_ratio1_{ratio1}_ru_{ru}_lam_{lam}_redist_{redist}_mr_{max_rounds}_seed_{seed}'
    elif mode == 'degree':
        m1_name = f'DBGS_{dataset_name}_eps_{eps}_ratio1_{ratio1}_ru_{ru}_lam_{lam}_redist_{redist}_mr_{max_rounds}_seed_{seed}'

    syn_graph_name = f"{m1_name}_gen.txt"
    eval_json_name = f"{m1_name}_eval.json"

    save_graph_flag = False
    max_depth = 0
    if mode == 'judge':
        max_depth = 10


    print("Loading dataset ...")
    t0 = time.time()

    graph = load_bipartite_graph(
        data_dir,
        dataset_path,
        sep
    )


    U_nodes = graph["U"]
    V_nodes = graph["V"]

    edges = graph["edges"]


    print(
        "Original graph:",
        len(U_nodes),
        len(V_nodes),
        len(edges)
    )

    if not os.path.exists(syn_graph_name):
        t1 = time.time()


        synthetic_graph = reconstruct_mod(
                    graph,
                    eps,
                    ratio1=ratio1,
                    ru=ru,
                    max_depth=max_depth,
                    max_rounds=max_rounds,
                    th_coeff=th_coeff,
                    eps_e=eps_e,
                    mode=mode,
                    redistribute=redist,
                    seed=seed
                    )

        t2 = time.time()
        print(f'Graph Generation Time:{t2-t1:.6f}s')

        synthetic_graph['U_reverse'] = graph['U_reverse']
        synthetic_graph['V_reverse'] = graph['V_reverse']

    
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if save_graph_flag:
            save_graph(
                synthetic_graph,
                output_path,
                syn_graph_name
            )
    else:
        print('Synthetic Graph exists')
        synthetic_graph = load_bipartite_graph(
                output_path,
                syn_graph_name,
                sep
            )


    print("\n******************************** Evaluation ********************************")

    cache_path = f"./results/{dataset_name}/orig_metrics.json"

    results = evaluate_all(
            graph,
            synthetic_graph,
            cache_path=cache_path
        )

    # for key, value in results.items():
    #     print(f"{key}: {value}")

    save_res_flag = True
    if save_res_flag:
        save_results(
        results,
        output_path,
        eval_json_name
        )

   


