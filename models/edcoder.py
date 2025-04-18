from itertools import chain
from functools import partial
import scipy.sparse as sp
import torch
import torch.nn as nn
import dgl
from dgl import DropEdge

from models.loss_func import sce_loss
from utils import create_norm
from models.han import HAN


def add_self_loop_to_isolated_nodes(g):
    in_degrees = g.in_degrees()
    isolated_nodes = torch.where(in_degrees == 0)[0]
    if len(isolated_nodes) > 0:
        g = dgl.add_edges(g, isolated_nodes, isolated_nodes)
    return g


class PhisHMAE_Model(nn.Module):
    def __init__(
        self,
        args,
        num_metapath: int,
        focused_feature_dim: int,
        mp2vec_feat_dim: int = None,
    ):
        super(PhisHMAE_Model, self).__init__()

        self.num_metapath = num_metapath
        self.focused_feature_dim = focused_feature_dim
        self.mp2vec_feat_dim = mp2vec_feat_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.activation = args.activation
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm
        self.feat_mask_rate = args.feat_mask_rate
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.loss_fn = args.loss_fn  # sce
        self.enc_dec_input_dim = self.focused_feature_dim

        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        # num head: encoder
        if self.encoder_type in ("gat", "dotgat", "han"):
            enc_num_hidden = self.hidden_dim // self.num_heads
            enc_nhead = self.num_heads
        else:
            enc_num_hidden = self.hidden_dim
            enc_nhead = 1

        # num head: decoder
        if self.decoder_type in ("gat", "dotgat", "han"):
            dec_num_hidden = self.hidden_dim // self.num_out_heads
            dec_nhead = self.num_out_heads
        else:
            dec_num_hidden = self.hidden_dim
            dec_nhead = 1
        dec_in_dim = self.hidden_dim

        # encoder
        self.encoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.encoder_type,
            enc_dec="encoding",
            in_dim=self.enc_dec_input_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=self.num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
        )

        # decoder
        self.decoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=self.enc_dec_input_dim,
            num_layers=1,
            nhead=enc_nhead,
            nhead_out=dec_nhead,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
            concat_out=True,
        )

        # Target Attribute Restoration (TAR)
        self.alpha_l = args.alpha_l
        self.attr_restoration_loss = self.setup_loss_fn(
            self.loss_fn,  # sce
            self.alpha_l,  # 3
        )
        self.__cache_gs = None
        self.enc_mask_token = nn.Parameter(
            torch.zeros(1, self.enc_dec_input_dim)
        )  # Define the mask token
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self._replace_rate = args.replace_rate
        self._leave_unchanged = args.leave_unchanged
        assert (
            self._replace_rate + self._leave_unchanged < 1
        ), "Replace rate + leave_unchanged must be smaller than 1"

        # Metapath-based Edge Reconstruction (MER)
        self.use_mp_edge_recon = args.use_mp_edge_recon
        self.mp_edge_recon_loss_weight = args.mp_edge_recon_loss_weight
        self.mp_edge_mask_rate = args.mp_edge_mask_rate
        self.mp_edge_alpha_l = args.mp_edge_alpha_l
        self.mp_edge_recon_loss = self.setup_loss_fn(
            self.loss_fn,  # sce
            self.mp_edge_alpha_l,  # 3
        )
        self.encoder_to_decoder_edge_recon = nn.Linear(
            dec_in_dim, dec_in_dim, bias=False
        )

        # Positional Feature Prediction (PFP)
        self.use_mp2vec_feat_pred = args.use_mp2vec_feat_pred
        self.mp2vec_feat_pred_loss_weight = args.mp2vec_feat_pred_loss_weight
        self.mp2vec_feat_alpha_l = args.mp2vec_feat_alpha_l
        self.mp2vec_feat_drop = args.mp2vec_feat_drop
        self.mp2vec_feat_pred_loss = self.setup_loss_fn(
            self.loss_fn, self.mp2vec_feat_alpha_l
        )
        self.pfp_mlp_decoder = nn.Sequential(
            nn.Linear(dec_in_dim, self.mp2vec_feat_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mp2vec_feat_dim, self.mp2vec_feat_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mp2vec_feat_dim, self.mp2vec_feat_dim),
        )

    @property
    def output_hidden_dim(self):
        return self.hidden_dim

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
        try:
            return float(input_mask_rate)
        except ValueError:
            if "~" in input_mask_rate:  # 0.6~0.8 Uniform sample
                mask_rate = [float(i) for i in input_mask_rate.split("~")]
                assert len(mask_rate) == 2
                if get_min:
                    return mask_rate[0]
                else:
                    return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
            elif "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                mask_rate = [float(i) for i in input_mask_rate.split(",")]
                assert len(mask_rate) == 3
                start_mask_rate = mask_rate[0]
                step_mask_rate = mask_rate[1]
                end_mask_rate = mask_rate[2]
                if get_min:
                    return min(start_mask_rate, end_mask_rate)
                else:
                    cur_mask_rate = start_mask_rate + epoch * step_mask_rate
                    if cur_mask_rate < min(
                        start_mask_rate, end_mask_rate
                    ) or cur_mask_rate > max(start_mask_rate, end_mask_rate):
                        return end_mask_rate
                    return cur_mask_rate
            else:
                raise NotImplementedError

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, feat, mps, **kwargs):
        # prepare for mp2vec feat pred
        if self.use_mp2vec_feat_pred:
            mp2vec_feat = feat[:, self.focused_feature_dim :]
            origin_feat = feat[:, : self.focused_feature_dim]
        else:
            origin_feat = feat

        # TAR
        dgl_graph_list = self.mps_to_dgl_graphs(mps)
        loss, feat_recon, att_mp, enc_out, mask_nodes = self.mask_attr_restoration(
            origin_feat, dgl_graph_list, kwargs.get("epoch", None)
        )

        # MER
        if self.use_mp_edge_recon:
            edge_recon_loss = self.mask_mp_edge_reconstruction(
                origin_feat, mps, kwargs.get("epoch", None)
            )
            loss += self.mp_edge_recon_loss_weight * edge_recon_loss

        # mp2vec feat pred
        if self.use_mp2vec_feat_pred:
            # MLP decoder
            mp2vec_feat_pred = self.pfp_mlp_decoder(enc_out)

            mp2vec_feat_pred_loss = self.mp2vec_feat_pred_loss(
                mp2vec_feat_pred, mp2vec_feat
            )

            loss += self.mp2vec_feat_pred_loss_weight * mp2vec_feat_pred_loss

        return loss, loss.item()

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        num_leave_nodes = int(self._leave_unchanged * num_mask_nodes)
        num_noise_nodes = int(self._replace_rate * num_mask_nodes)
        num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes
        token_nodes = mask_nodes[perm_mask[:num_real_mask_nodes]]
        noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
            :num_noise_nodes
        ]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        if num_noise_nodes > 0:
            out_x[noise_nodes] = x[noise_to_be_chosen]

        return out_x, (mask_nodes, keep_nodes)

    def mask_attr_restoration(self, feat, dgl_graph_list, epoch):
        cur_feat_mask_rate = self.get_mask_rate(self.feat_mask_rate, epoch=epoch)
        masked_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            feat, cur_feat_mask_rate
        )

        enc_out, _ = self.encoder(dgl_graph_list, masked_x, return_hidden=False)

        # ---- attribute reconstruction ----
        enc_out_mapped = self.encoder_to_decoder(enc_out)
        if self.decoder_type != "mlp":
            # re-mask
            enc_out_mapped[mask_nodes] = 0  # TODO: learnable? remove?

        if self.decoder_type == "mlp":
            feat_recon = self.decoder(enc_out_mapped)
        else:
            feat_recon, att_mp = self.decoder(dgl_graph_list, enc_out_mapped)

        x_init = feat[mask_nodes]
        x_rec = feat_recon[mask_nodes]
        loss = self.attr_restoration_loss(x_rec, x_init)

        return loss, feat_recon, att_mp, enc_out, mask_nodes

    def mask_mp_edge_reconstruction(self, feat, mps, epoch):
        # masked_ufu_adjacency_mx = sp.load_npz("/home/jxlu/project/PhishDetect/GraPhish/data/graphish/urls_in_html/urls_in_html_graphish_10_part/masked_ufu_1.npz")
        # masked_ufu_adjacency_mx = sparse_mx_to_torch_sparse_tensor(normalize_adj(masked_ufu_adjacency_mx))
        # mps[0] = masked_ufu_adjacency_mx.to(torch.device("cuda"))

        # masked_uwu_adjacency_mx = sp.load_npz(
        #     "/home/jxlu/project/PhishDetect/GraPhish/data/graphish/urls_in_html/urls_in_html_graphish_10_part/masked_uwu_1.npz"
        # )
        # masked_uwu_adjacency_mx = sparse_mx_to_torch_sparse_tensor(
        #     normalize_adj(masked_uwu_adjacency_mx)
        # )
        # mps[2] = masked_uwu_adjacency_mx.to(torch.device("cuda"))

        # masked_ufrfu_adjacency_mx = sp.load_npz("/home/jxlu/project/PhishDetect/GraPhish/data/graphish/urls_in_html/urls_in_html_graphish_10_part/masked_ufrfu_2.npz")
        # masked_ufrfu_adjacency_mx = sparse_mx_to_torch_sparse_tensor(normalize_adj(masked_ufrfu_adjacency_mx))
        # mps[1] = masked_ufrfu_adjacency_mx.to(torch.device("cuda"))

        masked_gs = self.mps_to_gs(mps)
        cur_mp_edge_mask_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)
        drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        for i in range(len(masked_gs)):
            masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i])  # we need to add self loop
        enc_rep, _ = self.encoder(masked_gs, feat, return_hidden=False)
        rep = self.encoder_to_decoder_edge_recon(enc_rep)

        # masked_gs = self.mps_to_dgl_graphs(masked_mps)
        # cur_mp_edge_mask_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)
        # drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        # for i in range(len(masked_gs)):
        #     if i != 1:
        #         masked_gs[i] = drop_edge(masked_gs[i])
        #     masked_gs[i] = add_self_loop_to_isolated_nodes(masked_gs[i])
        #     # masked_gs[i] = dgl.remove_self_loop(masked_gs[i])
        #     # masked_gs[i] = dgl.add_self_loop(masked_gs[i])  # we need to add self loop
        # enc_rep, _ = self.encoder(masked_gs, feat, return_hidden=False)
        # rep = self.encoder_to_decoder_edge_recon(enc_rep)

        if self.decoder_type == "mlp":
            feat_recon = self.decoder(rep)
        else:
            feat_recon, att_mp = self.decoder(masked_gs, rep)

        gs_recon = torch.mm(feat_recon, feat_recon.T)

        loss = None
        for i in range(len(mps)):
            if loss is None:
                loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
                # loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon_only_masked_places_list[i], mps_only_masked_places_list[i])  # loss only on masked places
            else:
                loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
                # loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon_only_masked_places_list[i], mps_only_masked_places_list[i])
        return loss

    def get_embeds(self, feat, mps, *args):
        if self.use_mp2vec_feat_pred:
            origin_feat = feat[:, : self.focused_feature_dim]
        else:
            origin_feat = feat
        gs = self.mps_to_dgl_graphs(mps)
        rep, _ = self.encoder(gs, origin_feat)
        return rep.detach()

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def mps_to_dgl_graphs(self, metapath_adjacency_matrices):
        if self.__cache_gs is None:
            dgl_graph_list = []
            mark = 0
            for mp in metapath_adjacency_matrices:
                node_cnt = mp.shape[0]
                indices = mp._indices()

                # if mark == 1:
                #     cur_graph = dgl.graph((indices[0], indices[1]),num_nodes=node_cnt)
                # else:
                #     cur_graph = dgl.graph((indices[0], indices[1]))

                cur_graph = dgl.graph((indices[0], indices[1]), num_nodes=node_cnt)

                # if mark == 2 or mark == 0:
                # if mark == 2:
                # cur_graph = dgl.remove_self_loop(cur_graph)
                cur_graph = add_self_loop_to_isolated_nodes(cur_graph)
                # cur_graph = dgl.add_self_loop(cur_graph)
                dgl_graph_list.append(cur_graph)
                mark = mark + 1
            return dgl_graph_list
        else:
            return self.__cache_gs

    def mps_to_gs(self, mps):
        if self.__cache_gs is None:
            gs = []
            for mp in mps:
                indices = mp._indices()
                cur_graph = dgl.graph((indices[0], indices[1]))
                gs.append(cur_graph)
            return gs
        else:
            return self.__cache_gs


def setup_module(
    m_type,
    num_metapath,
    enc_dec,
    in_dim,
    num_hidden,
    out_dim,
    num_layers,
    dropout,
    activation,
    residual,
    norm,
    nhead,
    nhead_out,
    attn_drop,
    negative_slope=0.2,
    concat_out=True,
) -> nn.Module:
    if m_type == "han":
        model = HAN(
            num_metapath=num_metapath,
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return model
