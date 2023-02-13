import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

import torch.nn.functional as F


class AUGRUCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, use_bias=True) -> None:
        super().__init__()

        input_size = input_size + hidden_size
        self.reset_gate = nn.Sequential(nn.Linear(input_size, hidden_size, bias=use_bias), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(input_size, hidden_size, bias=use_bias), nn.Sigmoid())
        self.h_hat_gate = nn.Sequential(nn.Linear(input_size, hidden_size, bias=use_bias), nn.Tanh())

    def forward(self, inputs, h_prev, attention_score=None):
        cat_inputs = torch.cat([inputs, h_prev], dim=1)
        r = self.reset_gate(cat_inputs)
        u = self.update_gate(cat_inputs)

        h_hat = self.h_hat_gate(torch.cat([inputs, r * h_prev], dim=1))
        score = attention_score.view(-1, 1) if attention_score is not None else torch.ones_like(u)
        u = score * u
        h_cur = (1. - u) * h_prev + u * h_hat
        return h_cur


class DebiasedGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, use_bias=True) -> None:
        super().__init__()

        input_size = input_size + hidden_size
        self.reset_gate = nn.Sequential(nn.Linear(input_size, hidden_size, bias=use_bias), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(input_size, hidden_size, bias=use_bias), nn.Sigmoid())
        self.h_hat_gate = nn.Sequential(nn.Linear(input_size, hidden_size, bias=use_bias), nn.Tanh())

    def forward(self, inputs, h_prev, attention_score=None):
        embedding_size = inputs.size(-1) // 2
        r_inputs = torch.cat([inputs, h_prev], dim=1)
        r = self.reset_gate(r_inputs)
        u_inputs = r_inputs.clone()
        # u_inputs[:, range(embedding_size, embedding_size + embedding_size)] = 1
        u = self.update_gate(u_inputs)

        h_hat = self.h_hat_gate(torch.cat([inputs, r * h_prev], dim=1))
        score = attention_score.view(-1, 1) if attention_score is not None else torch.ones(u.size(0), device=u.device)
        u = score * u
        h_cur = (1. - u) * h_prev + u * h_hat
        return h_cur


class DebiasedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru="AUGRU"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru == "AUGRU":
            self.rnn = AUGRUCell(input_size, hidden_size, bias)
        elif gru == "DeGRU":
            self.rnn = DebiasedGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs: PackedSequence, att_scores: PackedSequence = None, hidden_output=None):
        if not isinstance(inputs, PackedSequence) or (att_scores is not None and not isinstance(att_scores, PackedSequence)):
            raise NotImplementedError("DebiasedRNN only supports packed input and att_scores")

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores = att_scores.data if att_scores is not None else torch.ones(inputs.size(0), device=inputs.device)

        max_batch_size = int(batch_sizes[0])
        if hidden_output is None:
            hidden_output = torch.zeros(max_batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)

        outputs = torch.zeros(inputs.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device)

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                inputs[begin: begin + batch],
                hidden_output[0:batch],
                att_scores[begin: begin + batch],
            )
            outputs[begin: begin + batch] = new_hx
            hidden_output = new_hx
            begin += batch

        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices), None  # To match the output signature of nn.GRU


# class MultiHeadAttention(nn.Module):
#     def __init__(self, input_dim, attention_size, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.input_dim = input_dim
#         self.attention_size = attention_size
#         self.num_heads = num_heads
#
#         self.query = nn.Linear(input_dim, attention_size * num_heads)
#         self.key = nn.Linear(input_dim, attention_size * num_heads)
#         self.value = nn.Linear(input_dim, attention_size * num_heads)
#
#     def forward(self, inputs):
#         # linear projections for the queries, keys, and values
#         queries = self.query(inputs).view(-1, self.num_heads, self.attention_size).permute(0, 2, 1)
#         keys = self.key(inputs).view(-1, self.num_heads, self.attention_size).permute(0, 2, 1)
#         values = self.value(inputs).view(-1, self.num_heads, self.attention_size)
#
#         # calculate the attention weights
#         attention_weights = torch.bmm(queries, keys) / (self.attention_size ** 0.5)
#         attention_weights = F.softmax(attention_weights, dim=-1)
#
#         # calculate the attention-weighted inputs
#         attention_weighted_inputs = torch.bmm(attention_weights, values)
#         attention_weighted_inputs = attention_weighted_inputs.view(-1, self.input_dim)
#
#         return attention_weighted_inputs
#
#
# class AttentionGRUCell(nn.Module):
#     def __init__(self, input_dim, hidden_state_size, attention_size, num_heads):
#         super().__init__()
#         self.hidden_state_size = hidden_state_size
#         self.multi_head_attention = MultiHeadAttention(input_dim, attention_size, num_heads)
#         self.gru = nn.GRUCell(input_dim, hidden_state_size)
#
#     def forward(self, inputs, hidden_state):
#         attention_weighted_inputs = self.multi_head_attention(inputs)
#         updated_hidden_state = self.gru(attention_weighted_inputs, hidden_state)
#         return updated_hidden_state
