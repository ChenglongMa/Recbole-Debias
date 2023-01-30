import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_

from recbole_debias.model.abstract_recommender import DebiasedRecommender


class H2NET(SequentialRecommender):
    r"""
        DICE model, which equipped with DICESampler(in recbole-debias.sampler) and DICETrainer(in recbole-debias.trainer)
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # mcl: new arg
        self.n_users = dataset.num(self.USER_ID)
        #

        self.mask_field = config['MASK_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.dis_loss = config['dis_loss']
        self.dis_pen = config['dis_pen']
        self.int_weight = config['int_weight']
        self.pop_weight = config['pop_weight']
        self.adaptive = config['adaptive']

        # mcl: new params
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        # end of new params

        # define layers and loss
        self.users_int = nn.Embedding(self.n_users, self.embedding_size)
        self.items_int = nn.Embedding(self.n_items, self.embedding_size)  # TODO: padding_idx==0?
        self.users_pop = nn.Embedding(self.n_users, self.embedding_size)
        self.items_pop = nn.Embedding(self.n_items, self.embedding_size)  # TODO: padding_idx==0?

        # mcl: new embedding
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        # end of new embedding

        if self.dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif self.dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif self.dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        # parameters initialization
        self.apply(self._init_weights)

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

    def get_user_emb_total(self, user):

        user_emb = torch.cat((self.users_int.weight, self.users_pop.weight), 1)
        return user_emb[user]

    def get_item_emb_total(self, item):

        item_emb = torch.cat((self.items_int.weight, self.items_pop.weight), 1)
        return item_emb[item]

    def dcor(self, x, y):

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

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, item_seq, item_seq_len, factor):
        item_seq_emb = None
        if factor == 'int':
            # user_emb = self.users_int(user)
            item_seq_emb = self.items_int(item_seq)
        elif factor == 'pop':
            # user_emb = self.users_pop(user)
            item_seq_emb = self.items_pop(item_seq)
        elif factor == 'tot':
            # user_emb = self.get_user_emb_total(user)
            item_seq_emb = self.get_item_emb_total(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def forward2(self, user, item, factor):
        user_emb = None
        item_emb = None
        if factor == 'int':
            user_emb = self.users_int(user)
            item_emb = self.items_int(item)
        elif factor == 'pop':
            user_emb = self.users_pop(user)
            item_emb = self.items_pop(item)
        elif factor == 'tot':
            user_emb = self.get_user_emb_total(user)
            item_emb = self.get_item_emb_total(item)
        return torch.mul(user_emb, item_emb).sum(dim=1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item_p = interaction[self.ITEM_ID]
        item_n = interaction[self.NEG_ITEM_ID]
        mask = interaction[self.mask_field]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        score_p_int = self.forward(user, item_p, 'int')
        score_n_int = self.forward(user, item_n, 'int')
        score_p_pop = self.forward(user, item_p, 'pop')
        score_n_pop = self.forward(user, item_n, 'pop')

        score_p_total = score_p_int + score_p_pop
        score_n_total = score_n_int + score_n_pop

        loss_int = self.mask_bpr_loss(score_p_int, score_n_int, mask)
        loss_pop = self.mask_bpr_loss(score_n_pop, score_p_pop, mask) + self.mask_bpr_loss(score_p_pop, score_n_pop,
                                                                                           ~mask)
        loss_total = self.bpr_loss(score_p_total, score_n_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_emb_int = self.items_int(item_all)
        item_emb_pop = self.items_pop(item_all)
        user_all = torch.unique(user)
        user_emb_int = self.users_int(user_all)
        user_emb_pop = self.users_pop(user_all)
        dis_loss = self.criterion_discrepancy(user_emb_int, user_emb_pop) + self.criterion_discrepancy(item_emb_int,
                                                                                                       item_emb_pop)

        loss = loss_total + self.int_weight * loss_int + self.pop_weight * loss_pop - self.dis_pen * dis_loss
        return loss

    def adapt(self, decay):

        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item, 'tot')
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_emb_total(user)
        all_item_e = torch.cat((self.items_int.weight, self.items_pop.weight), 1)
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
