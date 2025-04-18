import argparse

datasets_args = {
    "phishscope_part0": {
        "type_num": [7039, 31164, 16597, 13157],
        "nei_num": 2,
        "n_labels": 3,
    },
    "phishscope_part1": {
        "type_num": [7033, 31317, 16679, 13280],
        "nei_num": 2,
        "n_labels": 3,
    },
    "phishscope_part2": {
        "type_num": [7024, 31569, 17502, 13427],
        "nei_num": 2,
        "n_labels": 3,
    },
    "phishscope_part3": {
        "type_num": [7015, 37781, 19791, 13067],
        "nei_num": 2,
        "n_labels": 3,
    },
    "phishscope_part4": {
        "type_num": [7006, 32589, 16706, 13246],
        "nei_num": 2,
        "n_labels": 3,
    },
}


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    # from GraphMAE
    parser.add_argument(
        "--num_heads", type=int, default=8, help="number of hidden attention heads"
    )
    parser.add_argument(
        "--num_out_heads", type=int, default=1, help="number of output attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of hidden layers"
    )
    parser.add_argument(
        "--residual", action="store_true", default=True, help="use residual connection"
    )
    parser.add_argument(
        "--feat_drop", type=float, default=0.4, help="input feature dropout"
    )
    parser.add_argument("--attn_drop", type=float, default=0, help="attention dropout")
    parser.add_argument("--norm", type=str, default="batchnorm")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--negative_slope",
        type=float,
        default=0.2,
        help="the negative slope of leaky relu for GAT",
    )
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument(
        "--feat_mask_rate",
        type=str,
        default="0.2,0.005,0.5",
        help="""The mask rate. If provide a float like '0.5', mask rate is static over the training. 
                        If provide two number connected by '~' like '0.4~0.9', mask rate is uniformly sampled over the training.
                        If Provide '0.7,-0.1,0.5', mask rate starts from 0.7, ends at 0.5 and reduce 0.1 for each epoch.""",
    )
    parser.add_argument(
        "--replace_rate",
        type=float,
        default=0.3,
        help="The replace rate. The ratio of nodes that is replaced by random nodes.",
    )
    parser.add_argument(
        "--leave_unchanged",
        type=float,
        default=0.1,
        help="The ratio of nodes left unchanged (no mask), but is asked to reconstruct.",
    )

    parser.add_argument("--encoder", type=str, default="han")
    parser.add_argument("--decoder", type=str, default="han")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument(
        "--alpha_l", type=float, default=3, help="pow index for sce loss"
    )
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=True)
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.99,
        help="decay the lr by gamma for ExponentialLR scheduler",
    )

    parser.add_argument(
        "--dataset", type=str, default="phishscope_part1"
    )
    parser.add_argument("--ratio", type=int, default=[10, 20, 40])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=512)  # 64
    parser.add_argument("--mae_epochs", type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument("--eva_lr", type=float, default=0.01)  # 0.05
    parser.add_argument("--eva_wd", type=float, default=0, help="weight decay")

    # The parameters of learning process
    parser.add_argument(
        "--patience", type=int, default=10
    )  # 5. we can change the value. this might impact the performance.
    parser.add_argument("--l2_coef", type=float, default=0)

    # The parameters of metapath2vec
    parser.add_argument(
        "--use_mp2vec_feat_pred",
        type=bool,
        default=True,
        help="Set to True to use the mp2vec feature regularization.",
    )
    parser.add_argument(
        "--mps_lr", type=float, default=0.005, help="mp2vec learning rate"
    )
    parser.add_argument("--mps_embedding_dim", type=int, default=64)
    parser.add_argument("--mps_walk_length", type=int, default=10)
    parser.add_argument("--mps_context_size", type=int, default=5)
    parser.add_argument("--mps_walks_per_node", type=int, default=3)
    parser.add_argument("--mps_num_negative_samples", type=int, default=3)
    parser.add_argument("--mps_batch_size", type=int, default=256)
    parser.add_argument("--mps_epoch", type=int, default=20)
    parser.add_argument("--mp2vec_feat_pred_loss_weight", type=float, default=0.5)
    parser.add_argument(
        "--mp2vec_feat_alpha_l",
        type=float,
        default=2,
        help="pow index for sce loss in edge reconstruction",
    )
    parser.add_argument(
        "--mp2vec_feat_drop", type=float, default=0.2, help="input feature dropout"
    )

    # read config
    parser.add_argument(
        "--use_cfg", action="store_true", help="Set to True to read config file"
    )

    # meta-path edge reconstruction
    parser.add_argument(
        "--use_mp_edge_recon",
        type=bool,
        default=True,
        help="Set to True to use the meta-path edge reconstruction.",
    )
    parser.add_argument("--mp_edge_recon_loss_weight", type=float, default=0.5)
    parser.add_argument(
        "--mp_edge_mask_rate",
        type=str,
        default="0.5",
        help="""The mask rate. If provide a float like '0.5', mask rate is static over the training. 
                        If provide two number connected by '~' like '0.4~0.9', mask rate is uniformly sampled over the training.
                        If Provide '0.7,-0.1,0.5', mask rate starts from 0.7, ends at 0.5 and reduce 0.1 for each epoch.""",
    )
    parser.add_argument(
        "--mp_edge_alpha_l",
        type=float,
        default=3,
        help="pow index for sce loss in edge reconstruction",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "clustering"],
    )

    parser.add_argument(
        "--n_labels",
        type=int,
        default=2,
        help="number of labels",
    )


    args, _ = parser.parse_known_args()
    for key, value in datasets_args[args.dataset].items():
        setattr(args, key, value)
    return args
