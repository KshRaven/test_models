
from TModels.util.qol import manage_params
from TModels.util.fancy_text import *

from torch import Tensor, device as DEVICE, dtype as DTYPE
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Vanilla
"""


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(TokenEmbedding, self).__init__()
        # BUILD
        self.embedding  = nn.Embedding(vocab_size, embed_size, device=device, dtype=dtype)

        # ATTRIBUTES
        self.embed_size = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, verbose=False):
        # Expands input to embedding space; [records, sequence] to [records, sequence, embed_size]
        tensor = self.embedding(tensor)
        if verbose:
            print(f"\nEmbedded Data =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class SinusoidalEncoding(nn.Module):
    def __init__(self, seq_length: int, embed_size: int, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(SinusoidalEncoding, self).__init__()
        # BUILD - [records, sequence, embed_size], EMBEDDING - [sequence, embed_size]
        self.positional_encoding = self._generate_encoding(seq_length, embed_size).\
            to(device=device, dtype=dtype)
        # EMBEDDING - [1, sequence, embed_size]
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

        # ATTRIBUTES
        self.max_seq_length = seq_length
        self.embed_size     = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, verbose: int = None):
        # Get the dimension shape of the input
        records, seq_length, embed_size = tensor.size()
        # Expanding positional encoding to shape of input
        positional_encoding = self.positional_encoding.expand(records, -1, embed_size)
        # Add encoding to tensor
        tensor = tensor + positional_encoding[:, :seq_length]
        if verbose:
            print(f"\nPositional Encoding =>\n{positional_encoding}\n\tdim = {positional_encoding.shape}")
            print(f"\nEncoded Sequences =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor

    @staticmethod
    def _generate_encoding(max_seq_length: int, embed_size: int, constant=10000.0):
        encoding = torch.zeros(max_seq_length, embed_size)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.multiply(
                torch.arange(0, embed_size, 2),
                (-torch.log(torch.tensor(constant)) / embed_size)
            )
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding


"""
Timeseries
"""


class BufferEmbedding(nn.Module):
    def __init__(self, inputs: int, embed_size: int, bias=False, convolve: dict[str, int] = None,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(BufferEmbedding, self).__init__()
        # BUILD
        if convolve is None:
            self.embedding  = nn.Linear(inputs, embed_size, bias=bias, device=device, dtype=dtype)
        else:
            if isinstance(convolve, bool) and convolve:
                convolve = {}
            self.kernel_size = manage_params(convolve, 'kernel_size', 3)
            self.stride = manage_params(convolve, 'stride', 1)
            self.padding = manage_params(convolve, 'padding', 1)
            self.padding_mode = manage_params(convolve, 'padding_mode', 'circular')
            self.embedding = nn.Conv1d(inputs, embed_size, self.kernel_size, self.stride, self.padding,
                                       bias=bias, padding_mode=self.padding_mode, device=device, dtype=dtype)

        # ATTRIBUTES
        self.embed_size = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, verbose: int = None):
        # Expand input to embedding space; [batch_size, sequence, features] to [batch_size, sequence, embed_size]
        if isinstance(self.embedding, nn.Linear):
            tensor = self.embedding(tensor).contiguous()
        else:
            tensor = self.embedding(tensor.mT).mT
        if verbose:
            print(f"\nEmbedded Data =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class BufferEncoding(nn.Module):
    def __init__(self, max_seq_len: int, embed_size: int, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(BufferEncoding, self).__init__()
        # BUILD
        self.source = torch.arange(max_seq_len, device=device, dtype=dtype)
        self.source = self.source.unsqueeze(0).unsqueeze(-1) / (max_seq_len-1)
        self.encode = nn.Linear(1, embed_size, device=device, dtype=dtype)
        self.actv = nn.SiLU()

        # ATTRIBUTES
        self.max_seq_len    = max_seq_len
        self.embed_size     = embed_size

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, pos_idx: int = None, verbose: int = None):
        batch_size, seq_len, embed_size = tensor.shape
        positions = self.source[:, :seq_len] if pos_idx is None else self.source[:, pos_idx:pos_idx+seq_len]
        positional_encoding = self.encode(positions).to(dtype=tensor.dtype).expand(batch_size, seq_len, embed_size)
        if self.actv is not None:
            positional_encoding = self.actv(positional_encoding)
        tensor = tensor + positional_encoding
        if verbose:
            if verbose >= 2:
                print(f"\nPositional Data =>\n{positional_encoding}\n\tdim = {positional_encoding.shape}")
            print(f"\nEncoded Data =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class RoPE(nn.Module):
    def __init__(self, max_seq_len: int, embed_size: int, heads: int, constant: int = 10000,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(RoPE, self).__init__()
        # BUILD - [batch_size, seq_len, head_dim / 2], EMBEDDING - [seq_len, head_dim / 2]
        self.complex_frequencies = self._generate_encoding(max_seq_len, embed_size // heads, constant, 0)
        self.complex_frequencies = self.complex_frequencies.to(device=device).unsqueeze(0).unsqueeze(-2)
        # EMBEDDING - [genomes, 1, sequence, embed_size]
        self.select = torch.arange(max_seq_len, device=device, dtype=torch.int32)

        # ATTRIBUTES
        self.max_seq_len = max_seq_len
        self.embed_size     = embed_size
        self.head_dim       = embed_size // heads
        self.constant       = constant

        # STATES
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    @staticmethod
    def _generate_encoding(seq_length: int, head_dim: int, constant: float = 10000.0, verbose: int = None):
        # Dimensions of embedding must be even
        assert head_dim % 2 == 0, f"Head dimension must be divisible by 2"
        # Get theta where theta_i = 10000 ^ (-2 * (i-1) / embedding) for i = [1, 2, ..., dim / 2]; [head_dim / 2]
        theta = 1.0 / torch.pow(constant, torch.arange(0, head_dim, 2).float() / head_dim)
        # Get positions as m; [sequence]
        positions   = torch.arange(seq_length)
        # Multiply theta by each position; [sequence] outer* [head_dim / 2] -> [sequence, head_dim / 2]
        angles      = torch.outer(positions, theta).float()
        # We compute complex number in polar form c = R * exp(i * m * theta); [sequence, head_dim / 2]
        complex_f   = torch.polar(torch.ones_like(angles), angles)
        if verbose:
            print(f"\nTheta =>\n{theta}\n\tdim = {theta.shape}")
            print(f"\nPositions =>\n{positions}\n\tdim = {positions.shape}")
            print(f"\nAngles =>\n{angles}\n\tdim = {angles.shape}")
            print(f"\nComplex Frequencies init =>\n{complex_f}\n\tdim = {complex_f.shape}")

        return complex_f

    def forward(self, tensor: Tensor, pos_idx: int = None, verbose: int = None):
        seq_len = tensor.shape[-3]
        # [batch_size, sequence, heads, head_dim] -> [batch_size, sequence, heads, head_dim]
        complex_tensor = torch.view_as_complex(tensor.view(*tensor.shape[:-1], -1, 2))
        # [batch_size, sequence, heads, head_dim] * [1, sequence, 1, head_dim / 2] = [batch_size, sequence, heads, head_dim / 2]
        complex_frequencies = self.complex_frequencies
        if seq_len > 1:
            complex_frequencies = torch.index_select(complex_frequencies, -3, self.select[:seq_len])
        else:
            if pos_idx is None:
                pos_idx = 0
            complex_frequencies = torch.index_select(complex_frequencies, -3, self.select[pos_idx:pos_idx+seq_len])
        rotated_tensor = complex_tensor * complex_frequencies
        # [batch_size, sequence, heads, head_dim / 2] -> [batch_size, sequence, heads, head_dim / 2, 2]
        split_tensor = torch.view_as_real(rotated_tensor)
        # [records, sequence, heads, head_dim / 2, 2] -> [records, sequence, heads, head_dim]
        # [records, sequence, heads, head_dim] -> [records, sequence, embed_size]
        tensor = split_tensor.reshape(*tensor.shape).type_as(tensor)
        if verbose and verbose >= 3:
            print(f"\nComplex Tensor=>\n{complex_tensor}\n\tdim = {complex_tensor.shape}")
            print(f"\nComplex Frequencies =>\n{complex_frequencies}\n\tdim = {complex_frequencies.shape}")
            print(f"\nRotated Tensor =>\n{rotated_tensor}\n\tdim = {rotated_tensor.shape}")
            print(f"\nSplit Tensor =>\n{split_tensor}\n\tdim = {split_tensor.shape}")
            print(f"\nEncoded Tensor =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class AttentionLambda(nn.Module):
    def __init__(self, heads: int, head_dim: int, layer_idx: int = None, init_mean=0., init_std=0.1,  epsilon=1e-8,
                 device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(AttentionLambda, self).__init__()
        if layer_idx is None:
            layer_idx = 0

        # BUILD
        self.q1 = nn.Parameter(torch.zeros(heads, head_dim, device=device, dtype=dtype).normal_(init_mean, init_std))
        self.q2 = nn.Parameter(torch.zeros(heads, head_dim, device=device, dtype=dtype).normal_(init_mean, init_std))
        self.k1 = nn.Parameter(torch.zeros(heads, head_dim, device=device, dtype=dtype).normal_(init_mean, init_std))
        self.k2 = nn.Parameter(torch.zeros(heads, head_dim, device=device, dtype=dtype).normal_(init_mean, init_std))
        self.init = 0.8 - 0.6 * np.exp(-0.3 * layer_idx)
        self.eps = epsilon

    def forward(self, q: Tensor = None, k: Tensor = None):
        # query:     (batch_size, q_len, heads, head_dim)
        # key:       (batch_size, k_len, heads, head_dim)
        # attention: (batch_size, heads, q_len, k_len)
        return (
            torch.exp(torch.sum(self.q1 * self.k1, -1)) - torch.exp(torch.sum(self.q2 * self.k2, -1)) + self.init
            # torch.exp(self.q1 * self.k1) - torch.exp(self.q2 * self.k2) + self.init
        ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # .unsqueeze(-1)


class Attention(nn.Module):
    def __init__(
            self, max_seq_len: int, embed_size: int, heads: int = None, kv_heads: int = None, differential=True,
            layer_idx: int = None, constant=10000.0, eps=1e-8, affine=True, causal_mask=True, bias=False,
            device: DEVICE = 'cpu', dtype: DTYPE = torch.float32, **options):
        super(Attention, self).__init__()
        if heads is None:
            heads = 1
        if embed_size % heads != 0:
            raise ValueError(f"Embedding dimensions must be a multiple of heads num")
        if kv_heads is None:
            kv_heads = heads
        if heads % kv_heads != 0:
            raise ValueError(f"Query heads num must be a multiple of number of Key-Value heads num")

        inputs: int  = manage_params(options, 'inputs', None)
        outputs: int = manage_params(options, 'outputs', None)

        # ATTRIBUTES
        self.heads      = heads
        self.embed_size = embed_size
        self.head_dim   = embed_size // heads
        self.kv_heads   = kv_heads
        self.q_kv_ratio = heads // kv_heads
        self.max_seq_len = max_seq_len
        self.differential = differential
        self.causal_mask = causal_mask
        self.constant   = constant

        # BUILD
        self.mult = 2 if differential else 1
        self.query_proj = nn.Linear(embed_size if not inputs else inputs, heads*self.head_dim*self.mult, bias, device, dtype)
        self.key_proj   = nn.Linear(embed_size if not inputs else inputs, kv_heads*self.head_dim*self.mult, bias, device, dtype)
        self.value_proj = nn.Linear(embed_size if not inputs else inputs, kv_heads*self.head_dim, bias, device, dtype)
        self.out_proj   = nn.Linear(embed_size, embed_size if not outputs else outputs, bias, device, dtype)
        self.rotary_embedding = RoPE(max_seq_len, embed_size * self.mult, heads, constant, device, dtype)
        self.softmax    = nn.Softmax(-1)
        self.norm       = nn.RMSNorm(self.head_dim, eps, affine, device, dtype)
        self.diff_lambda = AttentionLambda(heads, self.head_dim, layer_idx, 0.0, 0.1, eps, device, dtype) if differential else None

        self.k_cache = torch.zeros(1, max_seq_len, kv_heads, self.head_dim * self.mult, device=device, dtype=dtype)
        self.v_cache = torch.zeros(1, max_seq_len, kv_heads, self.head_dim, device=device, dtype=dtype)

        # STATES
        self.device = device
        self.dtype  = dtype

    def adjust_cache_size(self, batch_size: int):
        cache_size = self.k_cache.shape[0]
        padding = max(0, batch_size - cache_size)
        if padding > 0:
            self.k_cache = torch.cat([self.k_cache, torch.zeros(padding, *self.k_cache.shape[1:], device=self.device, dtype=self.dtype)], 0)
            self.v_cache = torch.cat([self.v_cache, torch.zeros(padding, *self.v_cache.shape[1:], device=self.device, dtype=self.dtype)], 0)

    def repeat_kv(self, tensor: Tensor):
        batch_size, seq_len, kv_heads, head_dim = tensor.shape
        if self.q_kv_ratio == 1:
            return tensor
        else:
            return tensor.unsqueeze(-2).expand(batch_size, seq_len, kv_heads, self.q_kv_ratio, head_dim).\
                reshape(batch_size, seq_len, kv_heads * self.q_kv_ratio, head_dim)

    def attention(self, query: Tensor, key: Tensor, value: Tensor, mask: bool, differential: bool, verbose: int = None):
        if differential:
            query, key = query.view(*query.shape[:-1], 2, -1), key.view(*key.shape[:-1], 2, -1)
        # Get the attention score (energy)
        energy = torch.einsum("bqhd,bkhd->bhqk" if not differential else "bqhad,bkhad->bahqk", [query, key])
        # queries shape: (batch_size, query_len, heads, head_dim)
        # key shape:     (batch_size, key_len, heads, head_dim)
        # energy shape:  (batch_size, heads, query_len, key_len)
        if verbose:
            print(f"\n{cmod('Energy =>', Fore.LIGHTYELLOW_EX)}\n{energy}, \n\tdim = {energy.shape}")

        if mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask_ = torch.ones_like(energy, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            energy.masked_fill_(mask_, -torch.inf)
            if verbose and verbose >= 2:
                print(f"\n{cmod('Mask =>', Fore.CYAN)}\n{mask_}, \n\tdim = {mask_.shape}")
                print(f"\n{cmod('Masked Energy =>', Fore.LIGHTYELLOW_EX)}\n{energy}, \n\tdim = {energy.shape}")

        # Get the softmax of the energy
        scores = self.softmax(energy / np.sqrt(self.head_dim))

        if differential:
            diff_lambda = self.diff_lambda()
            if verbose and verbose >= 2:
                print(f"\n{cmod('Lambda =>', Fore.LIGHTYELLOW_EX)}"
                      f"\n{torch.round(diff_lambda, decimals=4)}\n\tdim = {diff_lambda.shape}")
            scores = scores[:, 0] - (diff_lambda * scores[:, 1])

            if mask:
                scores.masked_fill_(torch.ones_like(scores, dtype=torch.bool).triu(1), -torch.inf)
                if verbose and verbose >= 3:
                    print(f"\n{cmod('Differential Masked Energy =>', Fore.LIGHTYELLOW_EX)}"
                          f"\n{energy}\n\tdim = {energy.shape}")
            scores = self.softmax(scores)

        if verbose:
            print(f"\n{cmod('Attention Scores =>', Fore.LIGHTYELLOW_EX)}"
                  f"\n{torch.round(scores, decimals=4)}\n\tdim = {scores.shape}")

        # Get the weighted sum of the values and reshape to remove heads
        attention = self.norm(torch.einsum("bhqv,bvhd->bqhd", [scores, value]))
        # scores shape:    (batch_size, heads, query_len, value_len)
        # values shape:    (batch_size, value_len, heads, head_dim)
        # attention shape: (batch_size, query_len, heads, head_dim) then concat last 2 dim
        if self.differential:
            attention = attention * (1 - self.diff_lambda.init)

        return scores, attention

    def forward(self, tensor: Tensor, context: Tensor = None, pos_idx: int = None, no_caching=False, verbose: int = None, get=False):
        if pos_idx is not None:
            pos_idx = self.max_seq_len + pos_idx if pos_idx < 0 else pos_idx
            assert 0 < pos_idx < self.max_seq_len
        if verbose:
            print(f'\n{cmod("Executing Self Attention", Fore.LIGHTBLUE_EX)}')

        # Linearize Q, K, V
        query: Tensor   = self.query_proj(tensor if context is None else context)
        key: Tensor     = self.key_proj(tensor)
        value: Tensor   = self.value_proj(tensor)
        if verbose:
            print(cmod('Post Linearization =>'))
            print(f"\n{cmod('Query =>', Fore.LIGHTYELLOW_EX)}\n{cmod(query, Fore.LIGHTRED_EX)}, \n\tdim = {query.shape}")
            print(f"\n{cmod('Key =>', Fore.LIGHTYELLOW_EX)}\n{cmod(key, Fore.LIGHTGREEN_EX)}, \n\tdim = {key.shape}")
            print(f"\n{cmod('Value =>', Fore.LIGHTYELLOW_EX)}\n{cmod(value, Fore.LIGHTBLUE_EX)}, \n\tdim = {value.shape}")

        batch_size, q_seq_len = query.shape[:2]

        # Reshape Q, K, V for each rep head
        query   = query.view(batch_size, -1, self.heads, self.head_dim*self.mult)
        key     = key.view(batch_size, -1, self.kv_heads, self.head_dim*self.mult)
        value   = value.view(batch_size, -1, self.kv_heads, self.head_dim)
        if verbose:
            print(f"\n{cmod('Q after Reshaping =>', Fore.LIGHTYELLOW_EX)}\n{query}, \n\tdim = {query.shape}")
            print(f"\n{cmod('K after Reshaping =>', Fore.LIGHTYELLOW_EX)}\n{key}, \n\tdim = {key.shape}")

        # ROTARY EMBEDDING
        query = self.rotary_embedding(query, pos_idx, verbose)
        key   = self.rotary_embedding(key, None, verbose)
        if verbose:
            print(f"\n{cmod('Q after Rotary Embedding =>', Fore.LIGHTYELLOW_EX)}\n{query}, \n\tdim = {query.shape}")
            print(f"\n{cmod('K after Rotary Embedding =>', Fore.LIGHTYELLOW_EX)}\n{key}, \n\tdim = {key.shape}")

        # UPDATE CACHE
        if pos_idx is not None and not no_caching:
            # (batch_size, seq_len, kv_heads, head_dim)
            if self.k_cache.shape[0] < batch_size:
                self.adjust_cache_size(batch_size)

            # SET
            start, stop = max(0, pos_idx-1), pos_idx+q_seq_len
            assert start >= 0 and stop <= self.max_seq_len
            self.k_cache[:batch_size, start:stop] = key
            self.v_cache[:batch_size, start:stop] = value

            # GET
            key   = self.k_cache[:batch_size, :stop]
            value = self.v_cache[:batch_size, :stop]

        # Duplicate K and V for kv heads num per query head
        key   = self.repeat_kv(key)
        value = self.repeat_kv(value)
        if verbose:
            print(f"\n{cmod('Duplicated K =>', Fore.LIGHTYELLOW_EX)}\n{key}, \n\tdim = {key.shape}")
            print(f"\n{cmod('Duplicated V =>', Fore.LIGHTYELLOW_EX)}\n{value}, \n\tdim = {value.shape}")

        scores, attention = self.attention(query, key, value, self.causal_mask and context is None,
                                           self.differential, verbose)
        attention = attention.reshape(batch_size, -1, self.embed_size)
        # out_view shape:  (batch_size, query_len, embed_size)
        if verbose:
            print(f"\n{cmod('Attented Values =>', Fore.LIGHTYELLOW_EX)}\n{attention}\n\tdim = {attention.shape}")

        # Apply weights
        tensor: Tensor = self.out_proj(attention)
        if verbose:
            print(f"\n{cmod('Output Projection =>', Fore.LIGHTCYAN_EX)}\n{tensor}\n\tdim = {tensor.shape}")

        if not get:
            return tensor
        else:
            return tensor, (scores, attention)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, embed_size: int, fwd_exp: int = None, bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32):
        super(SwiGLUFeedForward, self).__init__()
        if fwd_exp is None:
            fwd_exp = 4
        # hidden_size = 4 * embed_size
        # hidden_size = int(2 * hidden_size / 3)
        # if fwd_exp is not None:
        #     hidden_size = int(fwd_exp * hidden_size)
        # hidden_size = mult * ((hidden_size + mult - 1) // mult)
        hidden_size = fwd_exp * embed_size

        # BUILD
        self.w1 = nn.Linear(embed_size, hidden_size, bias, device, dtype)
        self.w2 = nn.Linear(hidden_size, embed_size, bias, device, dtype)
        self.w3 = nn.Linear(embed_size, hidden_size, bias, device, dtype)
        self.actv = nn.SiLU()

    def forward(self, tensor: Tensor):
        # (batch_size, seq_len, embed_size) -> (batch_size, seq_len, hidden_size)
        swish_ = self.actv(self.w1(tensor))
        # (batch_size, seq_len, embed_size) -> (batch_size, seq_len, hidden_size)
        tensor_ = self.w3(tensor)
        # (batch_size, seq_len, hidden_size) * (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        tensor = swish_ * tensor_
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, embed_size / out)
        tensor = self.w2(tensor)
        return tensor


class TransformerBlock(nn.Module):
    def __init__(
            self, seq_len: int, embed_size: int, heads: int = None, kv_heads: int = None, fwd_exp=4, differential=True,
            layer_idx: int = None, constant=10000.0, eps=1e-8, affine=True, causal_mask=True, dropout: int = None,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32,):
        super(TransformerBlock, self).__init__()
        if dropout is None:
            dropout = 0.0

        # BUILD
        self.attention   = Attention(seq_len, embed_size, heads, kv_heads, differential, layer_idx,
                                     constant, eps, affine, causal_mask, bias, device, dtype)
        self.att_norm    = nn.RMSNorm(embed_size, eps, affine, device, dtype)
        self.feedforward = SwiGLUFeedForward(embed_size, fwd_exp, bias, device, dtype)
        self.ffd_norm    = nn.RMSNorm(embed_size, eps, affine, device, dtype)
        self.dropout     = nn.Dropout(dropout)

        # ATTRIBUTES
        self.embed_size = embed_size
        self.heads      = heads
        self.epsilon    = eps
        self.head_dim   = embed_size // heads

        # STATES
        self.attention_tensor: Tensor = None
        self.attented_value: Tensor = None
        self.device: DEVICE = device
        self.dtype: DTYPE   = dtype

    def forward(self, tensor: Tensor, context: Tensor = None, pos_idx: int = None, no_caching=False, verbose: int = None, get=False):
        # Normalize then get the attention
        attention = self.attention(
                self.att_norm(tensor), context=context, pos_idx=pos_idx, no_caching=no_caching, verbose=verbose, get=get
        )
        if get:
            attention, (self.attention_tensor, self.attented_value) = attention
        # Apply residual connection then dropout
        tensor      = self.dropout(attention + (tensor if context is None else context))
        # Pass through feed forward
        activation  = self.feedforward(self.ffd_norm(tensor))
        # Apply residual connection
        tensor      = self.dropout(activation + tensor)
        if verbose:
            print(f"\n{cmod('Transformer Block Output =>', Fore.LIGHTYELLOW_EX)}\n{tensor}\n\tdim = {tensor.shape}")

        return tensor


class TransformerBase(nn.Module):
    def __init__(
            self, seq_len: int, embed_size: int, layers: int, heads: int = None, kv_heads: int = None, fwd_exp=4, 
            differential=True, constant=10000.0, eps=1e-8, affine=True, causal_mask=True, dropout: int = None,
            bias=False, device: DEVICE = 'cpu', dtype: DTYPE = torch.float32,):
        super(TransformerBase, self).__init__()
        if dropout is None:
            dropout = 0

        # BUILD
        self.layers: list[TransformerBlock] = nn.ModuleList()
        for layer_idx in range(layers):
            self.layers.append(
                TransformerBlock(
                    seq_len, embed_size, heads, kv_heads, fwd_exp, differential, layer_idx,
                    constant, eps, affine, causal_mask, dropout, bias, device, dtype
                )
            )

        # ATTRIBUTES
        self.seq_len    = seq_len
        self.embed_size = embed_size
        self.heads      = heads
        self.layer_num  = layers
        self.fwd_exp    = fwd_exp
        self.kv_heads   = kv_heads
        self.dropout    = dropout

        # STATE
        self.attention_tensors: list[Tensor] = None
        self.attented_values: list[Tensor] = None
        self.device     = device
        self.dtype      = dtype

    def forward(self, tensor: Tensor, context: Tensor = None, pos_idx: int = None, no_caching=False,
                verbose: int = None, get=False, single=False):
        if single:
            no_caching = True
        # Pass through the encoder blocks
        for layer_idx, layer in enumerate(self.layers):
            single_fetch = single and layer_idx == len(self.layers) - 1
            # shape (batch_size, seq_len, embed_size)
            if single_fetch:
                # Using last token index in sequence to get the next token
                set_context = torch.select(tensor, -2, -1).unsqueeze(-2)
            else:
                set_context = context
            tensor = layer(tensor, context=set_context, pos_idx=pos_idx, no_caching=no_caching,
                           verbose=verbose if layer_idx == 0 else False, get=get)

        return tensor

    def get_attention(self):
        return [(layer.attention_tensor, layer.attented_value) for layer in self.layers]
