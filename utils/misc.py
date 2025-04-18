import dgl
import torch
import numpy as np


def mps_to_dgl_graphs(metapath_adjacency_matrices):
    dgl_graph_list = []
    for mp in metapath_adjacency_matrices:
        node_cnt = mp.shape[0]
        indices = mp._indices()
        cur_graph = dgl.graph((indices[0], indices[1]), num_nodes=node_cnt)
        cur_graph = add_self_loop_to_isolated_nodes(cur_graph)
        dgl_graph_list.append(cur_graph)
    return dgl_graph_list


def add_self_loop_to_isolated_nodes(g):
    in_degrees = g.in_degrees()
    isolated_nodes = torch.where(in_degrees == 0)[0]
    if len(isolated_nodes) > 0:
        g = dgl.add_edges(g, isolated_nodes, isolated_nodes)
    return g


def mps_to_random_dgl_graphs(metapath_adjacency_matrices):
    dgl_graph_list = []
    node_cnt = metapath_adjacency_matrices[0].shape[0]

    original_ids = np.arange(node_cnt)
    shuffled_ids = np.arange(node_cnt)
    np.random.shuffle(shuffled_ids)

    node_mapping = {
        int(original): int(shuffled)
        for original, shuffled in zip(original_ids, shuffled_ids)
    }

    for mp in metapath_adjacency_matrices:
        indices = mp._indices()
        src_nodes, dst_nodes = indices[0], indices[1]

        # 这里执行非常慢的原因是使用了列表推导式和循环处理每个节点
        # 使用向量化操作可以显著提高性能
        src_nodes_np = src_nodes.cpu().numpy()
        dst_nodes_np = dst_nodes.cpu().numpy()

        # 使用numpy的向量化映射操作
        new_src_nodes_np = np.array(
            [node_mapping[int(node)] for node in src_nodes_np], dtype=np.int64
        )
        new_dst_nodes_np = np.array(
            [node_mapping[int(node)] for node in dst_nodes_np], dtype=np.int64
        )

        # 转回torch tensor并放到正确的设备上
        new_src_nodes = torch.tensor(
            new_src_nodes_np, dtype=src_nodes.dtype, device=src_nodes.device
        )
        new_dst_nodes = torch.tensor(
            new_dst_nodes_np, dtype=dst_nodes.dtype, device=dst_nodes.device
        )

        cur_graph = dgl.graph((new_src_nodes, new_dst_nodes), num_nodes=node_cnt)
        cur_graph = add_self_loop_to_isolated_nodes(cur_graph)
        dgl_graph_list.append(cur_graph)

    reverse_node_mapping = {
        shuffled: original for original, shuffled in node_mapping.items()
    }

    return dgl_graph_list, reverse_node_mapping
