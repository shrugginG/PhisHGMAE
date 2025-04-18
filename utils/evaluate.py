import numpy as np
import torch
from utils.logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score


def evaluate_cluster(embeds, y, n_labels, kmeans_random_state):
    Y_pred = (
        KMeans(n_labels, random_state=kmeans_random_state).fit(embeds).predict(embeds)
    )
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    return nmi, ari


def evaluate(
    embeds,
    idx_train,
    idx_val,
    idx_test,
    label,
    nb_classes,
    device,
    lr,
    wd,
    dataset=None,
    isTest=True,
    train_index=None,
):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for i in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average="macro")
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average="micro")

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average="macro")
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average="micro")

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)

        y_true = test_lbls.detach().cpu().numpy()
        y_score = best_proba[:, 1].detach().cpu().numpy()

        # ----
        # Save classification results to CSV
        # if i == 0:
        #     import pandas as pd
        #
        #     predict_label = torch.argmax(best_logits, dim=1).detach().cpu().numpy()
        #     is_correct = (predict_label == y_true).astype(int)
        #
        #     results_df = pd.DataFrame(
        #         {
        #             "index": idx_test.cpu().numpy(),
        #             "best_proba": y_score,
        #             "y_true": y_true,
        #             "predict_label": predict_label,
        #             "is_correct": is_correct,
        #         }
        #     )
        #
        #     # Read URL nodes file to get URLs
        #     url_nodes_df = pd.read_csv(
        #         f"/home/jxlu/project/PhishDetect/PhisHGMAE/data/{GRAPHISH_NAME}/{dataset.split('_graphish')[0].split(f'{GRAPHISH_NAME}_')[1]}/{dataset.split(f'{GRAPHISH_NAME}_')[1].replace('graphish','phishscope')}/url_nodes.csv"
        #     )
        #
        #     # Map index to URL by merging dataframef
        #     results_df = results_df.merge(
        #         url_nodes_df[["url_id", "url", "registered_domain"]],
        #         left_on="index",
        #         right_on="url_id",
        #         how="left",
        #     )
        #
        #     # Drop url_id column before saving
        #     results_df = results_df.drop("url_id", axis=1)
        #
        #     # Save to CSV file
        #     results_df.to_csv(
        #         f"/home/jxlu/project/PhishDetect/PhisHGMAE/result/{dataset}-{train_index}_result.csv",
        #         index=False,
        #     )

        # NOTE - For binary classification, we only need the probability of the positive class
        if nb_classes <= 2:
            auc_score = roc_auc_score(
                y_true=y_true,
                y_score=y_score,
            )
        else:
            auc_score = roc_auc_score(
                y_true=test_lbls.detach().cpu().numpy(),
                y_score=best_proba.detach().cpu().numpy(),
                multi_class="ovr",
            )

        auc_score_list.append(auc_score)

    if isTest:
        print(
            "\t[Classification] Macro-F1: [{:.4f}, {:.4f}]  Micro-F1: [{:.4f}, {:.4f}]  auc: [{:.4f}, {:.4f}]".format(
                np.mean(macro_f1s),
                np.std(macro_f1s),
                np.mean(micro_f1s),
                np.std(micro_f1s),
                np.mean(auc_score_list),
                np.std(auc_score_list),
            )
        )
        return np.mean(macro_f1s), np.mean(micro_f1s), np.mean(auc_score_list)
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)
