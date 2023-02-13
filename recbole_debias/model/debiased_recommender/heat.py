import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender

from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import ContextSeqEmbLayer, MLPLayers, SequenceAttLayer
from recbole.model.loss import BPRLoss
import torch.nn.functional as F
from recbole.utils import InputType, FeatureType
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from recbole_debias.model.abstract_recommender import DebiasedRecommender
from recbole_debias.model.debiased_recommender.h2net import InterestExtractorNetwork
from recbole_debias.model.layers import AUGRUCell, DebiasedRNN


class HEAT(SequentialRecommender):
    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
                Given users, calculate the scores between users and all candidate items.

                Args:
                    interaction (Interaction): Interaction class of the batch.

                Returns:
                    torch.Tensor: Predicted scores for given users and all candidate items,
                    shape: [n_batch_users * n_candidate_items]
                """
        raise NotImplementedError

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # get field names and parameter value from config
        self.device = config["device"]
        self.alpha = config["alpha"]
        self.gru = config["gru_type"]
        self.pooling_mode = config["pooling_mode"]
        self.dropout_prob = config["dropout_prob"]
        self.LABEL_FIELD = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.NEG_ITEM_SEQ = config["NEG_PREFIX"] + self.ITEM_SEQ
        self.NEG_ITEM_MASK_SEQ = config['MASK_FIELD']

        # causal embedding
        self.dis_loss = config['dis_loss']
        self.dis_pen = config['dis_pen']
        self.int_weight = config['int_weight']
        self.soc_weight = config['soc_weight']
        self.adaptive = config['adaptive']

        # prefix = 'train' if self.training else 'eval'
        # batch_size = config[f'{prefix}_batch_size']
        # neg_sample_num = config[f'{prefix}_neg_sample_args']['sample_num']
        # # See general_dataloader#_init_batch_size_and_step()#L50
        # self.n_pos_instance = max(batch_size // (1 + neg_sample_num), 1)

        if self.dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif self.dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif self.dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor
        # end of causal embedding

        self.types = ["user", "item"]
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

        num_item_feature = sum(
            1 if dataset.field2type[field] not in [FeatureType.FLOAT_SEQ, FeatureType.FLOAT] or field in config["numerical_features"] else 0
            for field in self.item_feat.interaction.keys()
        )
        num_user_feature = sum(
            1 if dataset.field2type[field] not in [FeatureType.FLOAT_SEQ, FeatureType.FLOAT] or field in config["numerical_features"] else 0
            for field in self.user_feat.interaction.keys()
        )
        user_feat_dim = 2 * num_user_feature * self.embedding_size  # mcl: concat interest embedding and social embedding, thus add `2 * `
        item_feat_dim = 2 * num_item_feature * self.embedding_size  # mcl: concat interest embedding and social embedding, thus add `2 * `
        mask_mat = (
            torch.arange(self.max_seq_length).to(self.device).view(1, -1)
        )  # init mask

        # init sizes of used layers
        self.att_list = [4 * item_feat_dim] + self.mlp_hidden_size
        self.interest_mlp_list = [2 * item_feat_dim] + self.mlp_hidden_size + [1]
        self.dnn_mlp_list = [2 * item_feat_dim + user_feat_dim] + self.mlp_hidden_size

        # init interest extractor layer, interest evolving layer embedding layer, MLP layer and linear layer
        self.interest_extractor = InterestExtractorNetwork(
            item_feat_dim, item_feat_dim, self.interest_mlp_list
        )
        # self.interest_evolution = InterestEvolvingLayer(
        #     mask_mat, item_feat_dim, item_feat_dim, self.att_list, gru=self.gru
        # )
        self.attention = Attention(item_feat_dim)
        self.interest_embedding_layer = ContextSeqEmbLayer(
            dataset, self.embedding_size, self.pooling_mode, self.device
        )

        self.social_embedding_layer = ContextSeqEmbLayer(
            dataset, self.embedding_size, self.pooling_mode, self.device
        )

        self.dnn_mlp_layers = MLPLayers(self.dnn_mlp_list, activation="Dice", dropout=self.dropout_prob, bn=True)
        self.dnn_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.apply(self._init_weights)
        self.other_parameter_name = ["interest_embedding_layer", "social_embedding_layer"]

    @staticmethod
    def _init_weights(module):
        """
        mcl: extend `recbole.model.init.xavier_normal_initialization`
        """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight)
            if module.bias is not None:
                constant_(module.bias, 0)

    @staticmethod
    def mask_bpr_loss(p_score, n_score, mask):
        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    @staticmethod
    def dcor(x, y):
        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(y[:, None] - y, p=2, dim=2)

        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    @staticmethod
    def calc_embedding_score(user_emb, pos_item_seq_emb, neg_item_seq_emb, item_seq_len):
        # Size: [batch_size, max_seq_len, emb_size] -> [batch_size, max_seq_len + 1, emb_size] -> [batch_size * (max_seq_len + 1), emb_size]
        # pos_item_emb = torch.cat((pos_item_seq_emb, target_item_emb.unsqueeze(dim=1)), dim=1).flatten(end_dim=-2)
        # neg_item_emb = torch.cat((neg_item_seq_emb, target_item_emb.unsqueeze(dim=1)), dim=1).flatten(end_dim=-2)
        # TODO: change max_len to exact length of valid embedding, may not necessary
        repeats = pos_item_seq_emb.size(1)
        # repeats = item_seq_len
        # a.repeat_interleave(torch.tensor([2, 3], device=a.device), dim=0)
        # Repeat user embedding to be the same shape of item_seq_emb
        user_emb = user_emb.repeat_interleave(repeats, dim=0)
        # Size: [batch_size * max_len, embedding_size]
        pos_item_emb = pos_item_seq_emb.flatten(end_dim=-2)
        neg_item_emb = neg_item_seq_emb.flatten(end_dim=-2)

        # Size: [batch_size * (max_seq_len + 1)]
        # pos_score = torch.mul(user_emb.unsqueeze(dim=1).expand_as(pos_item_seq_emb), pos_item_seq_emb).sum(dim=-1)
        # neg_score = torch.mul(user_emb.unsqueeze(dim=1).expand_as(neg_item_seq_emb), neg_item_seq_emb).sum(dim=-1)

        # Size: [batch_size * max_len]
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=-1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=-1)
        return pos_score, neg_score

    def get_embeddings(self, user, item_seq, neg_item_seq, next_items, embedding_layer):
        max_length = item_seq.shape[1]
        # concatenate the history item seq with the target item to get embedding together
        item_seq_next_item = torch.cat((item_seq, neg_item_seq, next_items.unsqueeze(1)), dim=-1)
        sparse_embedding, dense_embedding = embedding_layer(user, item_seq_next_item)
        # concat the sparse embedding and float embedding
        feature_table = {}
        for tp in self.types:
            feature_table[tp] = []
            if sparse_embedding[tp] is not None:
                feature_table[tp].append(sparse_embedding[tp])
            if dense_embedding[tp] is not None:
                feature_table[tp].append(dense_embedding[tp])

            feature_table[tp] = torch.cat(feature_table[tp], dim=-2)
            feature_table[tp] = feature_table[tp].flatten(start_dim=-2)

        # shape: [batch_size, emb_size_sum]
        user_feat_list = feature_table["user"]
        # shape of pos and neg feat_list: [batch_size, max_length, emb_size_sum]
        item_feat_list, neg_item_feat_list, target_item_feat_emb = feature_table[
            "item"
        ].split([max_length, max_length, 1], dim=1)
        target_item_feat_emb = target_item_feat_emb.squeeze(1)
        return user_feat_list, item_feat_list, neg_item_feat_list, target_item_feat_emb

    def forward(self, user, item_seq, neg_item_seq, neg_item_mask_seq, item_seq_len, next_items):
        neg_item_mask = neg_item_mask_seq.flatten()

        # Interest embedding
        int_user_emb, pos_int_item_seq_emb, neg_int_item_seq_emb, target_int_item_emb \
            = self.get_embeddings(user, item_seq, neg_item_seq, next_items, self.interest_embedding_layer)

        pos_int_score, neg_int_score = self.calc_embedding_score(int_user_emb, pos_int_item_seq_emb, neg_int_item_seq_emb, item_seq_len)
        int_loss = self.mask_bpr_loss(pos_int_score, neg_int_score, neg_item_mask)

        # Social embedding
        soc_user_emb, pos_soc_item_seq_emb, neg_soc_item_seq_emb, target_soc_item_emb \
            = self.get_embeddings(user, item_seq, neg_item_seq, next_items, self.social_embedding_layer)

        pos_soc_score, neg_soc_score = self.calc_embedding_score(soc_user_emb, pos_soc_item_seq_emb, neg_soc_item_seq_emb, item_seq_len)
        soc_loss = self.mask_bpr_loss(neg_soc_score, pos_soc_score, neg_item_mask) + self.mask_bpr_loss(pos_soc_score, neg_soc_score, ~neg_item_mask)

        # pos_score, neg_score = pos_int_score + pos_soc_score, neg_int_score + neg_soc_score
        int_item_emb = torch.cat([pos_int_item_seq_emb, neg_int_item_seq_emb], dim=0)
        soc_item_emb = torch.cat([pos_soc_item_seq_emb, neg_soc_item_seq_emb], dim=0)
        dis_loss = self.criterion_discrepancy(int_user_emb, soc_user_emb) + self.criterion_discrepancy(int_item_emb, soc_item_emb)

        # interest
        pos_item_seq_emb = torch.cat([pos_int_item_seq_emb, pos_soc_item_seq_emb], dim=-1)
        neg_item_seq_emb = torch.cat([neg_int_item_seq_emb, neg_soc_item_seq_emb], dim=-1)
        interest, aux_loss = self.interest_extractor(pos_item_seq_emb, item_seq_len, neg_item_seq_emb)

        target_item_emb = torch.cat([target_int_item_emb, target_soc_item_emb], dim=-1)
        # evolution = self.interest_evolution(target_item_emb, interest, item_seq_len)

        evolution = self.attention(interest, target_item_emb)

        user_emb = torch.cat([int_user_emb, soc_user_emb], dim=-1)
        # inputs = torch.cat([evolution, target_int_item_emb, int_user_emb], dim=-1)
        inputs = torch.cat([evolution, target_item_emb, user_emb], dim=-1)
        # input the DNN to get the prediction score
        outputs = self.dnn_mlp_layers(inputs)
        preds = self.dnn_predict_layer(outputs)
        return preds.squeeze(1), aux_loss, int_loss, soc_loss, dis_loss

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        neg_item_seq = interaction[self.NEG_ITEM_SEQ]
        neg_item_mask_seq = interaction[self.NEG_ITEM_MASK_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        label = interaction[self.LABEL_FIELD]
        output, aux_loss, int_loss, soc_loss, dis_loss = self.forward(
            user, item_seq, neg_item_seq, neg_item_mask_seq, item_seq_len, next_items
        )
        loss = self.loss(output, label) + self.alpha * aux_loss + self.int_weight * int_loss + self.soc_weight * soc_loss - self.dis_pen * dis_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        neg_item_seq = interaction[self.NEG_ITEM_SEQ]
        neg_item_mask_seq = interaction[self.NEG_ITEM_MASK_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        scores, *_ = self.forward(user, item_seq, neg_item_seq, neg_item_mask_seq, item_seq_len, next_items)
        return self.sigmoid(scores)


class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_q = nn.Linear(input_dim, input_dim)

    def forward(self, query, key):
        b, s, e = query.size()
        key = key.unsqueeze(1).expand(-1, s, -1)

        query = self.W_q(query)
        key = self.W_k(key)

        attention_scores = torch.sum(query * key, dim=-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * query, dim=1)

        return weighted_sum
