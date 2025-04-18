import datetime
import sys
import time
import warnings
import os
from loguru import logger

import torch
from torch_geometric.nn.models import MetaPath2Vec

from models.edcoder import PhisHMAE_Model
from utils import (
    evaluate,
    load_best_configs,
    load_data,
    metapath2vec_train,
    preprocess_features,
    set_random_seed,
)
from utils.params import build_args

warnings.filterwarnings("ignore")

SUB_GRAPH_NUM = 5

# Configure loguru
log_path = "logs"
if not os.path.exists(log_path):
    os.makedirs(log_path)
# Remove default handler
logger.remove()
# Add console output handler
logger.add(sys.stderr, level="INFO")
# Add file output handler
logger.add(
    os.path.join(log_path, "PhisHGMAE_{time}.log"),
    rotation="500 MB",  # Auto-rotate when file size reaches 500MB
    retention="10 days",  # Keep logs for 10 days
    level="INFO",
    encoding="utf-8",
)


def main(args):
    set_random_seed(args.seed)
    (
        (
            url_feat,  # Target node feats
            metapath_adjacency_matrices,  # List of metapath adjacency matrices
            node_labels,  # target node labels
            classifier_train_index,  # index of training nodes, [[20 * nb_classes], [40 * nb_classes], [80 * nb_classes]]
            classifier_val_index,  # index of val nodes
            classifier_test_index,  # index of test nodes
        ),
        phish_graph,
        mp2vec_metapaths,
    ) = load_data(args.dataset, args.ratio)
    nb_classes = node_labels.shape[-1]
    url_feat_dim = url_feat.shape[1]

    metapath_num = len(metapath_adjacency_matrices)  # metapath nums
    logger.info("Dataset: {}", args.dataset)
    logger.info("The number of meta-paths: {}", metapath_num)

    if args.use_mp2vec_feat_pred:
        mp2vec_url_feats = []
        assert args.mps_embedding_dim > 0
        for mp2vec_metapath in mp2vec_metapaths:
            logger.info("{}", mp2vec_metapath)
            metapath_model = MetaPath2Vec(
                phish_graph.edge_index_dict,
                args.mps_embedding_dim,
                mp2vec_metapath,
                args.mps_walk_length,
                args.mps_context_size,
                args.mps_walks_per_node,
                args.mps_num_negative_samples,
                sparse=True,
            )
            metapath2vec_train(args, metapath_model, args.mps_epoch, args.device)
            mp2vec_url_feat = metapath_model("url").detach()

            # free up memory
            del metapath_model
            if args.device.type == "cuda":
                mp2vec_url_feat = mp2vec_url_feat.cpu()
                mp2vec_url_feats.append(mp2vec_url_feat)
                torch.cuda.empty_cache()
        mp2vec_feat = torch.cat(
            [torch.FloatTensor(preprocess_features(feat)) for feat in mp2vec_url_feats],
            dim=1,
        )
        url_feat = torch.hstack([url_feat, mp2vec_feat])

    # model
    graphish_model = PhisHMAE_Model(
        args, metapath_num, url_feat_dim, args.mps_embedding_dim * len(mp2vec_metapaths)
    )

    optimizer = torch.optim.Adam(
        graphish_model.parameters(), lr=args.lr, weight_decay=args.l2_coef
    )
    # scheduler
    if args.scheduler:
        logger.info("--- Use schedular ---")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.scheduler_gamma,  # default: 0.99
        )
    else:
        scheduler = None

    graphish_model.to(args.device)
    url_feat = url_feat.to(args.device)
    metapath_adjacency_matrices = [
        metapath_adjacency_matrice.to(args.device)
        for metapath_adjacency_matrice in metapath_adjacency_matrices
    ]
    node_labels = node_labels.to(args.device)
    classifier_train_index = [i.to(args.device) for i in classifier_train_index]
    classifier_val_index = [i.to(args.device) for i in classifier_val_index]
    classifier_test_index = [i.to(args.device) for i in classifier_test_index]

    cnt_wait = 0
    best_loss = 1e9
    best_epoch = 0

    starttime = datetime.datetime.now()
    best_model_state_dict = None
    for epoch in range(args.mae_epochs):
        graphish_model.train()
        optimizer.zero_grad()
        loss, loss_item = graphish_model(
            url_feat,
            metapath_adjacency_matrices,
            epoch=epoch,
        )
        logger.info(
            "Epoch: {}, loss: {}, lr: {:.6f}",
            epoch,
            loss_item,
            optimizer.param_groups[0]["lr"],
        )
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            cnt_wait = 0
            best_model_state_dict = graphish_model.state_dict()
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            logger.info("Early stopping!")
            break
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("The best epoch is: {}", best_epoch)
    graphish_model.load_state_dict(best_model_state_dict)
    graphish_model.eval()
    embeds = graphish_model.get_embeds(url_feat, metapath_adjacency_matrices)

    torch.save(
        embeds,
        f"./embeddings/{args.dataset}.pt",
    )

    # Calculate time spent on getting embeddings
    embedding_endtime = datetime.datetime.now()
    embedding_time = (embedding_endtime - starttime).seconds
    logger.info("Time spent on getting embeddings: {} s", embedding_time)

    if args.task == "classification":
        macro_score_list, micro_score_list, auc_score_list = [], [], []
        for i in range(len(classifier_train_index)):
            macro_score, micro_score, auc_score = evaluate(
                embeds,
                classifier_train_index[i],
                classifier_val_index[i],
                classifier_test_index[i],
                node_labels,
                nb_classes,
                args.device,
                args.eva_lr,
                args.eva_wd,
                args.dataset,
                train_index=i,
            )
            macro_score_list.append(macro_score)
            micro_score_list.append(micro_score)
            auc_score_list.append(auc_score)

    else:
        sys.exit("wrong args.task.")

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    logger.info("Total time: {} s", time)


if __name__ == "__main__":
    args = build_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        args.device = torch.device("cpu")

    if args.use_cfg:
        if args.task == "classification":
            config_file_name = "configs.yml"
        else:
            sys.exit(f"No available config file for task: {args.task}")
        args = load_best_configs(args, config_file_name)

    for index in range(SUB_GRAPH_NUM):
        args.dataset = f"phishscope_part{index}"
        logger.info("Dataset: {}", args.dataset)
        logger.info("\n{}", args)

        main(args)

        logger.info("Sleeping for 7 seconds...")
        time.sleep(7)
