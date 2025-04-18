import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import torch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import HeteroData


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def trans_phishgraph_to_pyg_graph(neigs):
    phishgraph_metadata = [
        [
            {"src": "url", "dst": "fqdn", "relation": "url_fqdn"},
            {"src": "fqdn", "dst": "url", "relation": "fqdn_url"},
        ],
        [
            {"src": "url", "dst": "word", "relation": "url_word"},
            {"src": "word", "dst": "url", "relation": "word_url"},
        ],
        [
            {
                "src": "fqdn",
                "dst": "registered_domain",
                "relation": "fqdn_registered_domain",
            },
            {
                "src": "registered_domain",
                "dst": "fqdn",
                "relation": "registered_domain_fqdn",
            },
        ],
    ]
    # Refer: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html?highlight=heterodata#torch_geometric.data.HeteroData
    d = defaultdict(dict)
    for mp_i, nei1 in enumerate(neigs):
        dst_array_concat = np.concatenate(nei1)
        src_array_concat = []
        for src_id, dst_array in enumerate(nei1):
            src_array_concat.extend([src_id] * len(dst_array))
        src_array_concat = np.array(src_array_concat)

        forward_src_name = phishgraph_metadata[mp_i][0]["src"]
        forward_dst_name = phishgraph_metadata[mp_i][0]["dst"]
        forward_relation = phishgraph_metadata[mp_i][0]["relation"]
        d[(forward_src_name, forward_relation, forward_dst_name)]["edge_index"] = (
            torch.LongTensor(np.vstack([src_array_concat, dst_array_concat]))
        )

        backward_src_name = phishgraph_metadata[mp_i][1]["src"]
        backward_dst_name = phishgraph_metadata[mp_i][1]["dst"]
        backward_relation = phishgraph_metadata[mp_i][1]["relation"]
        d[(backward_src_name, backward_relation, backward_dst_name)]["edge_index"] = (
            torch.LongTensor(np.vstack([dst_array_concat, src_array_concat]))
        )

    pyg_phishgraph = HeteroData(d)
    return pyg_phishgraph


def load_phishscope_dataset(dataset, ratio):

    dataset_path = f"./data/phishscope/{dataset}/"

    # Load and process url labels
    label = np.load(dataset_path + "labels.npy").astype("int32")
    label = encode_onehot(label)

    # Load url neighbors
    url_fqdn_neighs = np.load(dataset_path + "nei_f.npy", allow_pickle=True)
    url_word_neighs = np.load(dataset_path + "nei_w.npy", allow_pickle=True)
    fqdn_registered_domain_neighs = np.load(
        dataset_path + "nei_r.npy", allow_pickle=True
    )

    feat_u = np.load(dataset_path + "u_urlnet_feat.npy").astype("float32")

    # Load adjacency matrices
    ufu_adjacency_mx = sp.load_npz(dataset_path + "ufu.npz")
    ufrfu_adjacency_mx = sp.load_npz(dataset_path + "ufrfu.npz")
    uwu_adjacency_mx = sp.load_npz(dataset_path + "uwu.npz")

    # Load classifi train, test, val indices
    classifier_train_indices = [
        np.load(dataset_path + "train_" + str(i) + ".npy") for i in ratio
    ]
    classifier_test_indices = [
        np.load(dataset_path + "test_" + str(i) + ".npy") for i in ratio
    ]
    classifier_val_indices = [
        np.load(dataset_path + "val_" + str(i) + ".npy") for i in ratio
    ]

    # Convert to torch tensors
    label = torch.FloatTensor(label)

    url_fqdn_neighs = [torch.LongTensor(i) for i in url_fqdn_neighs]
    url_word_neighs = [torch.LongTensor(i) for i in url_word_neighs]
    fqdn_registered_domain_neighs = [
        torch.LongTensor(i) for i in fqdn_registered_domain_neighs
    ]

    feat_u = torch.FloatTensor(feat_u)

    ufu_adjacency_mx = sparse_mx_to_torch_sparse_tensor(normalize_adj(ufu_adjacency_mx))
    ufrfu_adjacency_mx = sparse_mx_to_torch_sparse_tensor(
        normalize_adj(ufrfu_adjacency_mx)
    )
    uwu_adjacency_mx = sparse_mx_to_torch_sparse_tensor(normalize_adj(uwu_adjacency_mx))

    classifier_train_indices = [torch.LongTensor(i) for i in classifier_train_indices]
    classifier_val_indices = [torch.LongTensor(i) for i in classifier_val_indices]
    classifier_test_indices = [torch.LongTensor(i) for i in classifier_test_indices]

    return (
        [url_fqdn_neighs, url_word_neighs, fqdn_registered_domain_neighs],
        feat_u,
        [ufu_adjacency_mx, ufrfu_adjacency_mx, uwu_adjacency_mx],
        label,
        classifier_train_indices,
        classifier_val_indices,
        classifier_test_indices,
    )


def load_data(dataset, ratio):

    graphish_data = load_phishscope_dataset(dataset, ratio)

    phish_graph = trans_phishgraph_to_pyg_graph(graphish_data[0])

    mp2vec_metapaths = [
        [
            ("url", "url_fqdn", "fqdn"),
            ("fqdn", "fqdn_url", "url"),
        ],
        [
            ("url", "url_fqdn", "fqdn"),
            ("fqdn", "fqdn_registered_domain", "registered_domain"),
            ("registered_domain", "registered_domain_fqdn", "fqdn"),
            ("fqdn", "fqdn_url", "url"),
        ],
        [
            ("url", "url_word", "word"),
            ("word", "word_url", "url"),
        ],
    ]
    return graphish_data[1:], phish_graph, mp2vec_metapaths
