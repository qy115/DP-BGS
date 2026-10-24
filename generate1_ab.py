import numpy as np
from collections import defaultdict

def build_subgraph_graph(block):
    U_nodes = list(block["U_nodes"])
    V_nodes = list(block["V_nodes"])
    edges = list(block["edges"])

    adj_U = defaultdict(set)
    adj_V = defaultdict(set)

    for u in U_nodes:
        adj_U[u] = set()

    for v in V_nodes:
        adj_V[v] = set()

    for u, v in edges:
        adj_U[u].add(v)
        adj_V[v].add(u)

    degree_U = [
        len(adj_U[u])
        for u in U_nodes
    ]

    degree_V = [
        len(adj_V[v])
        for v in V_nodes
    ]

    return {
        "U": U_nodes,
        "V": V_nodes,
        "edges": edges,
        "adj_U": adj_U,
        "adj_V": adj_V,
        "degree_U": degree_U,
        "degree_V": degree_V,
        "num_U": len(U_nodes),
        "num_V": len(V_nodes)
    }




def perturb_graph_degrees(
        graph,
        eps_tu,
        eps_tv,
        rng
):
    

    noisy_degree_U = {}

    for i, u in enumerate(graph["U"]):

        degree = graph["degree_U"][i]

        noisy_value = (
            degree
            + rng.laplace(
                0.0,
                1.0 / eps_tu
            )
        )

        noisy_value = int(
            np.rint(noisy_value)
        )

        noisy_value = max(
            0,
            min(
                noisy_value,
                graph["num_V"]
            )
        )

        noisy_degree_U[u] = noisy_value

    noisy_degree_V = {}

    for i, v in enumerate(graph["V"]):

        degree = graph["degree_V"][i]

        noisy_value = (
            degree
            + rng.laplace(
                0.0,
                1.0 / eps_tv
            )
        )

        noisy_value = int(
            np.rint(noisy_value)
        )

        noisy_value = max(
            0,
            min(
                noisy_value,
                graph["num_U"]
            )
        )

        noisy_degree_V[v] = noisy_value

    return (
        noisy_degree_U,
        noisy_degree_V
    )



    
def perturb_total_edges(
        graph,
        eps_e,
        rng
):
    

    true_edges = len(graph["edges"])

    noisy_edges = (
        true_edges
        + rng.laplace(
            0.0,
            1.0 / eps_e
        )
    )

    noisy_edges = int(
        np.rint(noisy_edges)
    )

    noisy_edges = max(
        0,
        noisy_edges
    )

    return noisy_edges


def split_by_noisy_degree(
        graph,
        noisy_degree_U,
        noisy_degree_V,
        th_u,
        th_v
):
   

    U_high = []
    U_low = []

    for u in graph["U"]:

        if noisy_degree_U[u] >= th_u:
            U_high.append(u)
        else:
            U_low.append(u)

    V_high = []
    V_low = []

    for v in graph["V"]:

        if noisy_degree_V[v] >= th_v:
            V_high.append(v)
        else:
            V_low.append(v)

    U_groups = {
        0: U_high,
        1: U_low
    }

    V_groups = {
        0: V_high,
        1: V_low
    }

    U_gid = {}

    for gid, nodes in U_groups.items():

        for u in nodes:
            U_gid[u] = gid

    V_gid = {}

    for gid, nodes in V_groups.items():

        for v in nodes:
            V_gid[v] = gid

    blocks = {}

    for i in range(2):

        for j in range(2):

            blocks[(i, j)] = {
                "U_nodes": list(U_groups[i]),
                "V_nodes": list(V_groups[j]),
                "edges": []
            }

    # Assign original edges to the corresponding block.
    for u, v in graph["edges"]:

        bid = (
            U_gid[u],
            V_gid[v]
        )

        blocks[bid]["edges"].append(
            (u, v)
        )

    return blocks



def perturb_block_degree(
        block,
        eps_su,
        eps_sv,
        rng
):
  

    degree_U = defaultdict(int)
    degree_V = defaultdict(int)

    for u, v in block["edges"]:

        degree_U[u] += 1
        degree_V[v] += 1

    noisy_degree_U = {}
    noisy_degree_V = {}



    for u in block["U_nodes"]:

        true_degree = degree_U[u]

        noisy_value = (
            true_degree
            + rng.laplace(
                0.0,
                1.0 / eps_su
            )
        )

        noisy_value = int(
            np.rint(noisy_value)
        )

        noisy_value = max(
            0,
            min(
                noisy_value,
                len(block["V_nodes"])
            )
        )

        noisy_degree_U[u] = noisy_value


    for v in block["V_nodes"]:

        true_degree = degree_V[v]

        noisy_value = (
            true_degree
            + rng.laplace(
                0.0,
                1.0 / eps_sv
            )
        )

        noisy_value = int(
            np.rint(noisy_value)
        )

        noisy_value = max(
            0,
            min(
                noisy_value,
                len(block["U_nodes"])
            )
        )

        noisy_degree_V[v] = noisy_value

    block["noisy_degree_U"] = noisy_degree_U
    block["noisy_degree_V"] = noisy_degree_V

    block["noisy_edges"] = (
        sum(noisy_degree_U.values())
        +
        sum(noisy_degree_V.values())
    ) / 2.0

    return block


# Recursively partition the graph


def collect_leaf_blocks(
        graph,
        eps,
        ratio1=0.5,
        ru=0.5,
        max_depth=10,
        depth=0,
        th_coeff=2,
        threshold_u=None,
        threshold_v=None,
        seed=None
):
  

    rng = np.random.default_rng(seed)

    ratio2 = 1.0 - ratio1
    rv = 1.0 - ru

    # Current-level privacy budget
 

    eps_s1_tu = (
        eps
        * ratio1
        * ru
    )

    eps_s1_tv = (
        eps
        * ratio1
        * rv
    )

    eps_s2_su = (
        eps
        * ratio2
        * ru
    )

    eps_s2_sv = (
        eps
        * ratio2
        * rv
    )

    if min(
        eps_s1_tu,
        eps_s1_tv,
        eps_s2_su,
        eps_s2_sv
    ) <= 0:

        raise ValueError(
            "All privacy budgets must be positive."
        )

   
    # Perturb current graph only for recursive partitioning
   

    partition_noisy_U, partition_noisy_V = \
        perturb_graph_degrees(
            graph,
            eps_s1_tu,
            eps_s1_tv,
            rng
        )

   
  

    if len(partition_noisy_U) > 0:

        current_mean_u = np.mean(
            list(
                partition_noisy_U.values()
            )
        )

    else:

        current_mean_u = 0.0

    if len(partition_noisy_V) > 0:

        current_mean_v = np.mean(
            list(
                partition_noisy_V.values()
            )
        )

    else:

        current_mean_v = 0.0

   
   

    if threshold_u is None:
        threshold_u = current_mean_u

    if threshold_v is None:
        threshold_v = current_mean_v

    print()
    print("=" * 75)
    print("Recursion depth:", depth)
    print("Current eps:", eps)

    print(
        "eps_s1_tu:",
        eps_s1_tu
    )

    print(
        "eps_s1_tv:",
        eps_s1_tv
    )

    print(
        "eps_s2_su:",
        eps_s2_su
    )

    print(
        "eps_s2_sv:",
        eps_s2_sv
    )

    print(
        "Inherited th_u:",
        threshold_u
    )

    print(
        "Inherited th_v:",
        threshold_v
    )

    print(
        "Current noisy-degree mean U:",
        current_mean_u
    )

    print(
        "Current noisy-degree mean V:",
        current_mean_v
    )

    print(
        f"{th_coeff} / eps_s2_su:",
        th_coeff / eps_s2_su
    )

    print(
        f"{th_coeff} / eps_s2_sv:",
        th_coeff / eps_s2_sv
    )

    
    # Split current graph
  

    blocks = split_by_noisy_degree(
        graph,
        partition_noisy_U,
        partition_noisy_V,
        threshold_u,
        threshold_v
    )

    for bid in [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]:

        block = blocks[bid]

        print(
            "Block",
            bid,
            ": U =",
            len(block["U_nodes"]),
            ", V =",
            len(block["V_nodes"]),
            ", original edges =",
            len(block["edges"])
        )

    
    # Decide whether B00 continues


    noise_large = (
        threshold_u < th_coeff / eps_s2_su
        or
        threshold_v < th_coeff / eps_s2_sv
    )

    depth_limit = (
        depth >= max_depth
    )

    # Empty B00 cannot be recursively divided.
    b00_empty = (
        len(blocks[(0, 0)]["U_nodes"]) == 0
        or
        len(blocks[(0, 0)]["V_nodes"]) == 0
    )

    should_stop = (
        noise_large
        or depth_limit
        or b00_empty
    )

   
    # All four blocks become leaf blocks.


    if should_stop:

        if depth_limit:

            print(
                "Stop recursion because "
                "maximum depth is reached."
            )

        elif b00_empty:

            print(
                "Stop recursion because "
                "B00 is empty."
            )

        else:

            print(
                "Stop recursion because "
                "noise influence is large."
            )

        leaf_blocks = {}

        for bid in [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1)
        ]:

            block = blocks[bid]

            block = perturb_block_degree(
                block,
                eps_s2_su,
                eps_s2_sv,
                rng
            )

            block["depth"] = depth
            block["block_id"] = bid

            leaf_id = (
                depth,
                bid
            )

            block["leaf_id"] = leaf_id

            leaf_blocks[
                leaf_id
            ] = block

            print(
                "Leaf block",
                leaf_id,
                ": noisy edges =",
                block["noisy_edges"]
            )

        return leaf_blocks

   
    # B01, B10, B11 become leaf blocks.
    # Only B00 recursively continues.
   

    print(
        "Continue recursion on B00."
    )

    leaf_blocks = {}

  
    # Direct leaf blocks


    for bid in [
        (0, 1),
        (1, 0),
        (1, 1)
    ]:

        block = blocks[bid]

        block = perturb_block_degree(
            block,
            eps_s2_su,
            eps_s2_sv,
            rng
        )

        block["depth"] = depth
        block["block_id"] = bid

        leaf_id = (
            depth,
            bid
        )

        block["leaf_id"] = leaf_id

        leaf_blocks[
            leaf_id
        ] = block

        print(
            "Leaf block",
            leaf_id,
            ": noisy edges =",
            block["noisy_edges"]
        )

    

    b00_graph = build_subgraph_graph(
        blocks[(0, 0)]
    )

    next_eps = (
        eps * ratio2
    )

    child_leaf_blocks = collect_leaf_blocks(
        b00_graph,
        next_eps,
        ratio1=ratio1,
        ru=ru,
        max_depth=max_depth,
        depth=depth + 1,
        th_coeff=th_coeff,

        # Current graph mean becomes the threshold for
        # the next recursive level.
        threshold_u=current_mean_u,
        threshold_v=current_mean_v,

        seed=int(
            rng.integers(
                0,
                2**32 - 1
            )
        )
    )

    for local_id, block in child_leaf_blocks.items():

        leaf_blocks[
            local_id
        ] = block

    return leaf_blocks



# Root-level global perturbation and first split decision


def collect_all_leaf_blocks_mod(
        graph,
        eps,
        ratio1=0.5,
        ru=0.5,
        max_depth=10,
        th_coeff=2,
        eps_e=0.001,
        mode='judge',
        seed=None
):
    

    if eps <= eps_e:

        raise ValueError(
            "eps must be larger than eps_e."
        )

    rng = np.random.default_rng(seed)

    ratio2 = 1.0 - ratio1
    rv = 1.0 - ru


    # Privacy budget for total edge count

    noisy_edges = perturb_total_edges(
        graph,
        eps_e,
        rng
    )

    # Remaining budget for degree / recursive mechanism.
    eps_graph = eps - eps_e

    if mode == 'degree' or mode == 'block':
        eps_graph = eps

    # Initial budget split

    eps_s1_tu = (
        eps_graph
        * ratio1
        * ru
    )

    eps_s1_tv = (
        eps_graph
        * ratio1
        * rv
    )

    eps_s2_su = (
        eps_graph
        * ratio2
        * ru
    )

    eps_s2_sv = (
        eps_graph
        * ratio2
        * rv
    )

    print()
    print("#" * 80)
    print("GLOBAL PERTURBATION")
    print("#" * 80)

    print(
        "True edges:",
        len(graph["edges"])
    )

    print(
        "Noisy edges:",
        noisy_edges
    )

    print(
        "eps_e:",
        eps_e
    )

    print(
        "Remaining graph budget:",
        eps_graph
    )

    
    # Threshold estimation
  

    if graph["num_U"] > 0:

        th_u = (
            noisy_edges
            / float(graph["num_U"])
        )

    else:

        th_u = 0.0

    if graph["num_V"] > 0:

        th_v = (
            noisy_edges
            / float(graph["num_V"])
        )

    else:

        th_v = 0.0

    print(
        "Estimated th_u:",
        th_u
    )

    print(
        "Estimated th_v:",
        th_v
    )

    print(
        "eps_s1_tu:",
        eps_s1_tu
    )

    print(
        "eps_s1_tv:",
        eps_s1_tv
    )

    print(
        "eps_s2_su:",
        eps_s2_su
    )

    print(
        "eps_s2_sv:",
        eps_s2_sv
    )

    # print(
    #     f"{th_coeff} / eps_s2_su:",
    #     th_coeff / eps_s2_su
    # )

    # print(
    #     f"{th_coeff} / eps_s2_sv:",
    #     th_coeff / eps_s2_sv
    # )


    # First partition decision
    

    need_partition = (
        th_u >= th_coeff / eps_s2_su
        and
        th_v >= th_coeff / eps_s2_sv
    )

    if mode == 'degree':
        need_partition = False
        print('Mode: degree')
    elif mode == 'block':
        need_partition = True
        print('Mode: block')

    # No partition.
   

    if not need_partition:

        print()
        print(
            "No block partition is required."
        )

        final_eps_tu = (
            eps_s1_tu
            + eps_s2_su
        )

        final_eps_tv = (
            eps_s1_tv
            + eps_s2_sv
        )

        print(
            "Final global eps_tu:",
            final_eps_tu
        )

        print(
            "Final global eps_tv:",
            final_eps_tv
        )

        # Final global degree perturbation.
   
        global_noisy_U, global_noisy_V = \
             perturb_graph_degrees(
                            graph,
                            final_eps_tu,
                            final_eps_tv,
                            rng)

        return (
            None,
            global_noisy_U,
            global_noisy_V,
            noisy_edges,
            False
        )


    # Partition is required.
    # First 2 x 2 partition is based on the global noisy
    # degrees.

    # First global perturbation
    # This perturbation is used only to estimate whether
    # partitioning is necessary.
 


    preliminary_U, preliminary_V = \
                 perturb_graph_degrees(
                                graph,
                                eps_s1_tu,
                                eps_s1_tv,
                                rng)

    # print()
    print(
        "Block partition is required."
    )

    blocks = split_by_noisy_degree(
        graph,
        preliminary_U,
        preliminary_V,
        th_u,
        th_v
    )

    for bid in [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1)
    ]:

        block = blocks[bid]

        print(
            "Root block",
            bid,
            ": U =",
            len(block["U_nodes"]),
            ", V =",
            len(block["V_nodes"]),
            ", original edges =",
            len(block["edges"])
        )


    # Check whether recursion should stop immediately.
  

    root_noise_large = (
        th_u < th_coeff / eps_s2_su
        or
        th_v < th_coeff / eps_s2_sv
    )

    root_depth_limit = (
        max_depth <= 0
    )

    if mode == 'block':
        root_depth_limit = True

    if root_noise_large or root_depth_limit:

        leaf_blocks = {}

        for bid in [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1)
        ]:

            block = perturb_block_degree(
                blocks[bid],
                eps_s2_su,
                eps_s2_sv,
                rng
            )

            block["depth"] = 0
            block["block_id"] = bid
            block["leaf_id"] = (
                0,
                bid
            )

            leaf_blocks[
                (0, bid)
            ] = block

        return (
            leaf_blocks,
            preliminary_U,
            preliminary_V,
            noisy_edges,
            True
        )

    # Continue recursively on B00.
    # B01/B10/B11 are immediately leaf blocks.


    leaf_blocks = {}

    for bid in [
        (0, 1),
        (1, 0),
        (1, 1)
    ]:

        block = perturb_block_degree(
            blocks[bid],
            eps_s2_su,
            eps_s2_sv,
            rng
        )

        block["depth"] = 0
        block["block_id"] = bid
        block["leaf_id"] = (
            0,
            bid
        )

        leaf_blocks[
            (0, bid)
        ] = block

    # B00 recursive processing
  

    b00_graph = build_subgraph_graph(
        blocks[(0, 0)]
    )

    child_eps = (
        eps_graph * ratio2
    )

    child_leaf_blocks = collect_leaf_blocks(
        b00_graph,
        child_eps,
        ratio1=ratio1,
        ru=ru,
        max_depth=max_depth,
        depth=1,
        th_coeff=th_coeff,

        # The threshold used at depth=1 is the mean noisy
        # degree of the larger ROOT graph.
        threshold_u=th_u,
        threshold_v=th_v,

        seed=int(
            rng.integers(
                0,
                2**32 - 1
            )
        )
    )

    for leaf_id, block in child_leaf_blocks.items():

        leaf_blocks[
            leaf_id
        ] = block

    return (
        leaf_blocks,
        preliminary_U,
        preliminary_V,
        noisy_edges,
        True
    )


# Redistribute global noisy degree to all leaf blocks


def redistribute_leaf_block_degrees(
        leaf_blocks,
        global_noisy_U,
        global_noisy_V
):
   

    U_to_blocks = defaultdict(list)

    for leaf_id, block in leaf_blocks.items():

        for u in block["U_nodes"]:

            U_to_blocks[u].append(
                leaf_id
            )

    for u, global_degree in global_noisy_U.items():

        related_blocks = U_to_blocks.get(
            u,
            []
        )

        if not related_blocks:
            continue

    
        # Original noisy block degrees

        values = []

        for leaf_id in related_blocks:

            value = leaf_blocks[
                leaf_id
            ]["noisy_degree_U"].get(
                u,
                0
            )

            values.append(
                max(0, int(value))
            )

        total = sum(values)


        if total <= 0:

            ratios = [
                1.0 / len(related_blocks)
                for _ in related_blocks
            ]

        else:

            ratios = [
                value / total
                for value in values
            ]


        global_degree = int(
            max(0, global_degree)
        )

        remaining = global_degree

        for i, leaf_id in enumerate(
            related_blocks
        ):

            if i == len(related_blocks) - 1:

                allocated = remaining

            else:

                allocated = int(
                    round(
                        global_degree
                        * ratios[i]
                    )
                )

                allocated = max(
                    0,
                    min(
                        allocated,
                        remaining
                    )
                )

                remaining -= allocated

            leaf_blocks[
                leaf_id
            ]["noisy_degree_U"][u] = allocated



    V_to_blocks = defaultdict(list)

    for leaf_id, block in leaf_blocks.items():

        for v in block["V_nodes"]:

            V_to_blocks[v].append(
                leaf_id
            )

    for v, global_degree in global_noisy_V.items():

        related_blocks = V_to_blocks.get(
            v,
            []
        )

        if not related_blocks:
            continue

        values = []

        for leaf_id in related_blocks:

            value = leaf_blocks[
                leaf_id
            ]["noisy_degree_V"].get(
                v,
                0
            )

            values.append(
                max(0, int(value))
            )

        total = sum(values)

        if total <= 0:

            ratios = [
                1.0 / len(related_blocks)
                for _ in related_blocks
            ]

        else:

            ratios = [
                value / total
                for value in values
            ]

        global_degree = int(
            max(0, global_degree)
        )

        remaining = global_degree

        for i, leaf_id in enumerate(
            related_blocks
        ):

            if i == len(related_blocks) - 1:

                allocated = remaining

            else:

                allocated = int(
                    round(
                        global_degree
                        * ratios[i]
                    )
                )

                allocated = max(
                    0,
                    min(
                        allocated,
                        remaining
                    )
                )

                remaining -= allocated

            leaf_blocks[
                leaf_id
            ]["noisy_degree_V"][v] = allocated

    # Recalculate noisy edge count of every leaf block


    for leaf_id, block in leaf_blocks.items():

        sum_U = sum(
            block["noisy_degree_U"].values()
        )

        sum_V = sum(
            block["noisy_degree_V"].values()
        )

        block["noisy_edges"] = (
            sum_U + sum_V
        ) / 2.0

    return leaf_blocks

# Initialize edges inside one leaf block


def initialize_block(
        block,
        seed=None
):
  

    rng = np.random.default_rng(seed)

    adj_U = defaultdict(set)
    adj_V = defaultdict(set)

    for u in block["U_nodes"]:
        adj_U[u] = set()

    for v in block["V_nodes"]:
        adj_V[v] = set()

    U_order = list(
        block["U_nodes"]
    )

    rng.shuffle(
        U_order
    )

    V_nodes = block["V_nodes"]

    if len(V_nodes) == 0:

        return {
            "adj_U": adj_U,
            "adj_V": adj_V
        }

    for u in U_order:

        target = int(
            block["noisy_degree_U"].get(
                u,
                0
            )
        )

        target = max(
            0,
            min(
                target,
                len(V_nodes)
            )
        )

        if target <= 0:
            continue

        selected = rng.choice(
            V_nodes,
            size=target,
            replace=False
        )

        for v in selected:

            adj_U[u].add(v)
            adj_V[v].add(u)

    return {
        "adj_U": adj_U,
        "adj_V": adj_V
    }


def correct_block(
        block,
        state,
        max_rounds=5,
        seed=None
):
    

    rng = np.random.default_rng(seed)

    adj_U = state["adj_U"]
    adj_V = state["adj_V"]

    for round_id in range(
        max_rounds
    ):

        positive = []
        negative = []

        # Calculate V degree difference.
  

        for v in block["V_nodes"]:

            current_degree = len(
                adj_V[v]
            )

            target_degree = int(
                block["noisy_degree_V"].get(
                    v,
                    0
                )
            )

            difference = (
                current_degree
                - target_degree
            )

            if difference > 0:

                positive.append(
                    (
                        v,
                        difference
                    )
                )

            elif difference < 0:

                negative.append(
                    (
                        v,
                        -difference
                    )
                )

        positive.sort(
            key=lambda x: x[1],
            reverse=True
        )

        negative.sort(
            key=lambda x: x[1],
            reverse=True
        )

        pair_num = min(
            len(positive),
            len(negative)
        )

        moved_edges = 0





        for i in range(pair_num):

            vp, excess = positive[i]
            vn, deficit = negative[i]

    

            candidates = [
                u
                for u in adj_V[vp]
                if vn not in adj_U[u]
            ]

            if not candidates:
                continue

            move_num = min(
                excess,
                deficit,
                len(candidates)
            )

            if move_num <= 0:
                continue

            if move_num == len(candidates):

                selected = candidates

            else:

                selected = list(
                    rng.choice(
                        candidates,
                        size=move_num,
                        replace=False
                    )
                )


            for u in selected:

                adj_U[u].remove(vp)
                adj_V[vp].remove(u)

                adj_U[u].add(vn)
                adj_V[vn].add(u)

                moved_edges += 1

      

        if moved_edges == 0:
            break

   

    edges = []

    for u in block["U_nodes"]:

        for v in adj_U[u]:

            edges.append(
                (u, v)
            )

    return {
        "U": list(block["U_nodes"]),
        "V": list(block["V_nodes"]),
        "edges": edges,
        "adj_U": adj_U,
        "adj_V": adj_V,
        "degree_U": [
            len(adj_U[u])
            for u in block["U_nodes"]
        ],
        "degree_V": [
            len(adj_V[v])
            for v in block["V_nodes"]
        ],
        "num_U": len(block["U_nodes"]),
        "num_V": len(block["V_nodes"])
    }


# Reconstruct ONE leaf block


def reconstruct_one_block(
        block,
        max_rounds,
        rng
):
  

    state = initialize_block(
        block,
        seed=int(
            rng.integers(
                0,
                2**32 - 1
            )
        )
    )

    graph = correct_block(
        block,
        state,
        max_rounds=max_rounds,
        seed=int(
            rng.integers(
                0,
                2**32 - 1
            )
        )
    )

    return graph


# Reconstruct graph directly from global noisy degrees


def reconstruct_global_graph(
        graph,
        noisy_degree_U,
        noisy_degree_V,
        max_rounds=50,
        seed=None
):
   

    rng = np.random.default_rng(seed)

    global_block = {
        "U_nodes": list(graph["U"]),
        "V_nodes": list(graph["V"]),
        "edges": list(graph["edges"]),

        "noisy_degree_U": dict(
            noisy_degree_U
        ),

        "noisy_degree_V": dict(
            noisy_degree_V
        )
    }

    reconstructed = reconstruct_one_block(
        global_block,
        max_rounds=max_rounds,
        rng=rng
    )

    return reconstructed

# Merge all reconstructed leaf blocks


def merge_graphs(
        graph_list
):
  

    U_set = set()
    V_set = set()
    edges = []

    for graph in graph_list:

        U_set.update(
            graph["U"]
        )

        V_set.update(
            graph["V"]
        )

        edges.extend(
            graph["edges"]
        )

    U_nodes = list(U_set)
    V_nodes = list(V_set)

    adj_U = defaultdict(set)
    adj_V = defaultdict(set)

    for u in U_nodes:
        adj_U[u] = set()

    for v in V_nodes:
        adj_V[v] = set()

    for u, v in edges:

        adj_U[u].add(v)
        adj_V[v].add(u)

    degree_U = [
        len(adj_U[u])
        for u in U_nodes
    ]

    degree_V = [
        len(adj_V[v])
        for v in V_nodes
    ]

    return {
        "U": U_nodes,
        "V": V_nodes,
        "edges": edges,
        "adj_U": adj_U,
        "adj_V": adj_V,
        "degree_U": degree_U,
        "degree_V": degree_V,
        "num_U": len(U_nodes),
        "num_V": len(V_nodes)
    }





# Complete adaptive two-stage reconstruction


def reconstruct_mod(
        graph,
        eps,
        ratio1=0.5,
        ru=0.5,
        max_depth=0,
        max_rounds=50,
        th_coeff=4.472,
        eps_e=0.001,
        mode='block',
        redistribute=True,
        seed=42
):
   

    print()
    print("#" * 80)
    print("STAGE 0: TOTAL EDGE PERTURBATION")
    print("#" * 80)

 

   


    (
            leaf_blocks,
            global_noisy_U,
            global_noisy_V,
            noisy_edges,
            used_blocks
        ) = collect_all_leaf_blocks_mod(
        graph,
        eps,
        ratio1=ratio1,
        ru=ru,
        max_depth=max_depth,
        th_coeff=th_coeff,
        eps_e=eps_e,
        mode=mode,
        seed=seed
    )

   
    # No block partition

    if not used_blocks:

        print()
        print("#" * 80)
        print("NO BLOCK PARTITION")
        print("#" * 80)

        print(
            "Noisy total edges:",
            noisy_edges
        )

        print(
            "Global noisy U degree sum:",
            sum(global_noisy_U.values())
        )

        print(
            "Global noisy V degree sum:",
            sum(global_noisy_V.values())
        )

        print()
        print(
            "Direct global reconstruction..."
        )

        synthetic_graph = reconstruct_global_graph(
            graph,
            global_noisy_U,
            global_noisy_V,
            max_rounds=max_rounds,
            seed=(
                seed + 100000
                if seed is not None
                else None
            )
        )

        print()
        print(
            "Final synthetic U nodes:",
            synthetic_graph["num_U"]
        )

        print(
            "Final synthetic V nodes:",
            synthetic_graph["num_V"]
        )

        print(
            "Final synthetic edges:",
            len(synthetic_graph["edges"])
        )

        return synthetic_graph

  
    # Block partition is used


    print()
    
    if redistribute:
        print("#" * 80)
        print("STAGE 2: GLOBAL DEGREE REDISTRIBUTION")
        print("#" * 80)

        leaf_blocks = redistribute_leaf_block_degrees(
            leaf_blocks,
            global_noisy_U,
            global_noisy_V
        )

      

        total_noisy_U = 0
        total_noisy_V = 0

        for leaf_id, block in leaf_blocks.items():

            block_U_sum = sum(
                block["noisy_degree_U"].values()
            )

            block_V_sum = sum(
                block["noisy_degree_V"].values()
            )

            total_noisy_U += block_U_sum
            total_noisy_V += block_V_sum

            print(
                "Leaf block:",
                leaf_id,
                "| U:",
                len(block["U_nodes"]),
                "| V:",
                len(block["V_nodes"]),
                "| noisy U degree sum:",
                block_U_sum,
                "| noisy V degree sum:",
                block_V_sum,
                "| noisy edges:",
                block["noisy_edges"]
            )

        print()
        print(
            "Noisy total edges:",
            noisy_edges
        )

        print(
            "Root noisy U degree sum:",
            sum(global_noisy_U.values())
        )

        print(
            "Final block U degree sum:",
            total_noisy_U
        )

        print(
            "Root noisy V degree sum:",
            sum(global_noisy_V.values())
        )

        print(
            "Final block V degree sum:",
            total_noisy_V
        )
    else:
        print("#" * 80)
        print("STAGE 2: NO GLOBAL DEGREE REDISTRIBUTION")
        print("#" * 80)

    # Reconstruct all leaf blocks
  

    print()
    print("#" * 80)
    print("STAGE 3: LEAF BLOCK RECONSTRUCTION")
    print("#" * 80)

    rng = np.random.default_rng(
        seed + 100000
        if seed is not None
        else None
    )

    reconstructed_graphs = []

    for leaf_id, block in leaf_blocks.items():

        print(
            "Reconstructing leaf block:",
            leaf_id
        )

        reconstructed = reconstruct_one_block(
            block,
            max_rounds=max_rounds,
            rng=rng
        )

        reconstructed_graphs.append(
            reconstructed
        )

        print(
            "  U nodes:",
            len(reconstructed["U"])
        )

        print(
            "  V nodes:",
            len(reconstructed["V"])
        )

        print(
            "  generated edges:",
            len(reconstructed["edges"])
        )

    # Merge


    print()
    print("#" * 80)
    print("STAGE 4: MERGE ALL LEAF BLOCKS")
    print("#" * 80)

    synthetic_graph = merge_graphs(
        reconstructed_graphs
    )

    print(
        "Final synthetic U nodes:",
        synthetic_graph["num_U"]
    )

    print(
        "Final synthetic V nodes:",
        synthetic_graph["num_V"]
    )

    print(
        "Final synthetic edges:",
        len(synthetic_graph["edges"])
    )

    return synthetic_graph