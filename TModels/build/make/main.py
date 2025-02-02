
from TModels.build import Model
from TModels.build.make.sub import *
from TModels.util.fancy_text import *
from TModels.util.qol import manage_params

from torch import Tensor, nn
from typing import Union

import torch


DATASET = tuple[Tensor, Tensor]


class Quantizer:
    def __init__(self, levels: int):
        self.levels: int = levels
        self.max: float = None
        self.min: float = None
        self.encoder: dict[Tensor, int] = dict()
        self.decoder: dict[int, Tensor] = dict()
        self.out_features: int = None
        self.classes_num: int = None

    def generate(self, datasets: Union[DATASET, list[DATASET]], extend=0.0, inp_norm=None, out_norm=None):
        if not isinstance(datasets, list):
            datasets = [datasets]

        max_ = 1.0
        min_ = 0.0
        for dataset in datasets:
            for form, tensor in enumerate(dataset):
                if form == 0 and inp_norm is not None:
                    tensor = inp_norm.pre_classifier(tensor)
                elif form == 1 and out_norm is not None:
                    tensor = out_norm.pre_classifier(tensor)
                max_ = max(torch.max(tensor).item(), max_)
                min_ = min(torch.min(tensor).item(), min_)
        self.max = max_ * (1 + extend)
        self.min = (min_ - (max_ * extend) if min_ == 0 else min_ * (1 + extend))

        index = 0
        self.encoder: dict[Tensor, int] = dict()
        self.decoder: dict[int, Tensor] = dict()
        self.out_features: int = None
        for dataset in datasets:
            outputs = dataset[1]
            if out_norm is not None:
                outputs = out_norm.pre_classifier(outputs)
            out_feat = outputs.shape[-1]
            if self.out_features is None:
                self.out_features = out_feat
            else:
                if out_feat != self.out_features:
                    raise ValueError(f"Number of output features do not match")
            for output_sequence in outputs:
                for activation in output_sequence:
                    # if activation not in self.encoder:
                    if not any(torch.all(activation == saved_actv) for saved_actv in self.encoder.keys()):
                        self.encoder[activation] = index
                        self.decoder[index] = activation
                        index += 1
        self.classes_num = index

    def initialized(self):
        return self.levels is not None

    def __call__(self, normalized_tensor: Tensor, dtype: DTYPE = torch.int32, clamp=True, debug=False):
        if self.initialized():
            if clamp:
                normalized_tensor = torch.clamp(normalized_tensor, self.min, self.max)
            div = (self.max - self.min) / self.levels
            quantized_tensor = torch.floor(normalized_tensor / div).to(dtype=dtype)
            if debug:
                print(f"Quantised tensor = {quantized_tensor[:3]}")
            return quantized_tensor
        else:
            raise RuntimeError(f"Quantizer has not been initialized")

    def generate_targets(self, tensor: Tensor, dtype: DTYPE = torch.long) -> Tensor:
        clone = tensor[:, :, 0].clone()
        for rec_idx, output_sequence in enumerate(tensor):
            for seq_idx, activation in enumerate(output_sequence):
                found = False
                for saved_activation, index in self.encoder.items():
                    if torch.all(activation == saved_activation):
                        clone[rec_idx, seq_idx] = index
                        found = True
                        break
                if not found:
                    raise ValueError(f"Activation '{activation}' (dtype={type(activation)}) has not been recorded in "
                                     f"self.encoder: {list(self.encoder.keys())}")
        return clone.to(dtype=dtype)

    def decode_predictions(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len = tensor.shape
        clone = tensor.unsqueeze(-1).expand(batch_size, seq_len, self.out_features).clone()
        for rec_idx, output_sequence in enumerate(clone):
            for seq_idx, index in enumerate(output_sequence[:, 0]):
                found = False
                for saved_index, activation in self.decoder.items():
                    if index.item() == saved_index:
                        try:
                            clone[rec_idx, seq_idx, :] = activation
                        except RuntimeError as e:
                            print(f"Tensor =>\n{clone}\n\tshape={clone.shape}, dtype={type(clone)}")
                            print(f"Activation =>\n{activation}\n\tshape={activation.shape}, dtype={type(activation)}")
                            raise e
                        found = True
                        break
                if not found:
                    raise ValueError(f"Index '{index}' (dtype={type(index)}) has not been recorded in "
                                     f"self.decoder: {list(self.decoder.keys())}")
        return clone


class Transformer(nn.Module):
    def __init__(
            self, inputs: int, outputs: int, embed_size: int, max_seq_len: int, layers: int,
            heads: int = None, kv_heads: int = None, differential=True, dropout: int = None,
            bias=False, device = torch.device('cpu'), dtype: torch.dtype = torch.float32, **options):
        super(Transformer, self).__init__()
        self.distribution       = manage_params(options, 'distribution', 'normal')
        self.fwd_exp            = manage_params(options, 'fwd_exp', None)
        self.epsilon            = manage_params(options, 'epsilon', 1e-5)
        self.constant           = manage_params(options, 'constant', 10000)
        self.affine             = manage_params(options, 'affine', True)
        self.causal_mask        = manage_params(options, 'causal_mask', True)
        self.primary_activation = manage_params(options, 'pri_actv', nn.SiLU())
        self.secondary_activation = manage_params(options, 'sec_actv', None)
        self.convolve           = manage_params(options, 'convolve', None)

        if dropout is None:
            dropout = 0

        # BUILD
        self.embedder    = BufferEmbedding(inputs, embed_size, bias, self.convolve, device, dtype)
        self.encoder     = BufferEncoding(max_seq_len, embed_size, device, dtype)
        self.transformer = TransformerBase(
            max_seq_len, embed_size, layers, heads, kv_heads, self.fwd_exp, differential,
            self.constant, self.epsilon, self.affine, self.causal_mask, dropout, bias, device, dtype
        )
        self.dec_norm   = nn.RMSNorm(embed_size, self.epsilon, self.affine, device, dtype)
        output_dim      = outputs if self.distribution != 'discrete' else 2 ** outputs
        self.decode     = nn.Linear(embed_size, output_dim, bias, device, dtype)
        self.dropout    = nn.Dropout(dropout)

        # STATE
        self.device = device
        self.dtype  = dtype
        self.eval()

        # ATTRIBUTES
        self.max_seq_len = max_seq_len

    def forward(self, tensor: Tensor, pos_idx: int = None, verbose: int = None, get=False, single=False):
        if verbose:
            print(f"\nTransformer Input =>\n{tensor}\n\tdim = {tensor.shape}")
        squeeze = tensor.ndim == (2 if self.distribution != 'discrete' else 1)
        if squeeze:
            tensor = tensor.unsqueeze(0)
        if pos_idx is not None:
            tensor = tensor[:, :pos_idx+1]

        tensor = self.embedder(tensor, verbose=verbose)
        if self.primary_activation is not None:
            tensor = self.primary_activation(tensor)
        tensor = self.dropout(self.encoder(tensor, verbose=verbose))
        tensor = self.transformer(tensor, pos_idx=pos_idx, verbose=verbose, get=get, single=single)
        tensor = self.decode(self.dec_norm(tensor))
        if self.secondary_activation is not None:
            tensor = self.secondary_activation(tensor)

        if squeeze:
            tensor = tensor.squeeze(0)
        if self.distribution == 'discrete':
            tensor = torch.argmax(tensor, -1)
        if verbose:
            print(f"\nTransformer Output =>\n{tensor}\n\tdim = {tensor.shape}")

        return tensor

    def get_attention(self):
        a, v = [], []
        for ai, vi in self.transformer.get_attention():
            a.append(ai)
            v.append(vi)
        return a, v

    def infer(self, inputs: Tensor, pos_idx: int = None, get=False, single=True, verbose=False):
        with torch.no_grad():
            if single:
                pos_idx = -1
            if pos_idx is not None:
                seq_len = inputs.shape[-2]
                pos_idx = seq_len + pos_idx if pos_idx < 0 else pos_idx
                assert 0 < pos_idx < seq_len
            outputs = self.forward(inputs, pos_idx, verbose, get, single)
            return outputs


class Reformer(Model):
    def __init__(
            self, inputs: int, pol_out: int, val_out: int, embed_size: int, max_seq_len: int, layers: int, heads: int,
            kv_heads: int = None, differential=True, dropout: float = 0.1, bias=False, feedback=False,
            device: torch.device = 'cpu', dtype: torch.dtype = torch.float32, **options):
        super(Reformer, self).__init__()
        self.pri_actv           = manage_params(options, 'pri_actv', nn.SiLU())
        self.sec_actv           = manage_params(options, 'sec_actv', nn.Tanh())
        self.affine             = manage_params(options, 'affine', True)
        self.distribution       = manage_params(options, ['distribution', 'dist'], 'normal')
        self.epsilon            = manage_params(options, 'epsilon', 1e-8)
        options['sec_actv'] = None
        self.probabilistic      = manage_params(options, ['prob', 'probabilistic'], False)
        feed_drop               = manage_params(options, 'feedback_dropout', None)
        self.bias_enabled       = bias

        # BUILD
        self.feedback_feat = (pol_out + val_out) if feedback else None
        # feedback_gain = [
        #     nn.Linear(pol_out+val_out, embed_size, device, dtype)
        # ] if feedback else None
        self.feedback_gain = nn.RMSNorm(pol_out+val_out, self.epsilon, self.affine, device, dtype) if feedback else None
        self.feedback_dropout = nn.Dropout(feed_drop if feed_drop else 0)
        self.pol_proj = Transformer(
            inputs, embed_size, embed_size, max_seq_len, layers, heads, kv_heads, differential, dropout,
            bias, device, dtype, **options,
        )
        self.mean_std = nn.Linear(embed_size, pol_out*(2 if self.distribution != 'discrete' else 1), bias, device, dtype)
        self.val_proj = Transformer(
            inputs, embed_size, embed_size, max_seq_len, layers, heads, kv_heads, differential, dropout,
            bias, device, dtype, **options,
        )
        self.decode   = nn.Linear(embed_size, val_out, bias, device, dtype)
        self.dropout  = nn.Dropout(dropout if dropout else 0)

        # STATE
        self.device = device
        self.dtype = dtype
        self.eval()

        # ATTRIBUTES
        self.seq_len    = max_seq_len
        self.embed_size = embed_size
        self.feedback   = self.feedback_gain is not None or self.feedback_dropout is not None
        self.obs_size   = inputs - (0 if not self.feedback else (pol_out+val_out))
        self.pol_size   = pol_out
        self.val_size   = val_out
        self.idx: int   = None
        self.single     = False

    def get_feedback(self, state: Tensor):
        if self.feedback:
            observation, feedback = torch.split(state, self.obs_size, -1)
            assert feedback.shape[-1] == (self.pol_size+self.val_size)
            if self.feedback_gain is not None:
                feedback = self.feedback_gain(feedback)
            if self.feedback_dropout is not None:
                feedback = self.feedback_dropout(feedback)
            state = torch.cat((observation, feedback), -1)
        return state

    def get_latent(self, model: Transformer, state: Tensor, idx: int = None, verbose=False, get=False, single=False):
        latent = model.forward(self.get_feedback(state), idx, verbose, get, single)
        if self.pri_actv is not None:
            latent = self.pri_actv(latent)
        return latent

    def get_mean(self, latent: Tensor) -> Tensor:
        mean = latent[..., :self.pol_size]
        if self.probabilistic:
            mean = F.tanh(mean)
        elif self.sec_actv is not None:
            mean = self.sec_actv(mean)
        return self.reduce(mean)

    def get_std(self, latent: Tensor) -> Tensor:
        if self.distribution != 'discrete':
            log_std = latent[..., self.pol_size:self.pol_size*2]
            if self.probabilistic:
                std = torch.exp(-9.21 + F.tanh(log_std) * 6.91)
            else:
                std = torch.exp(log_std)
                if self.sec_actv is not None:
                    std = self.sec_actv(std)
            return self.reduce(std)
        else:
            return None

    def get_action(self, state: Tensor) -> tuple[Tensor, Tensor]:
        latent      = self.get_latent(self.pol_proj, state, self.idx, single=self.single)
        source      = self.mean_std(latent)
        mean        = self.get_mean(source)
        std         = self.get_std(source)
        dist        = self.dist(mean, std, None)
        action      = dist.sample()
        log_prob    = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, state: Tensor, action: Tensor) -> [Tensor, Union[Tensor, None]]:
        latent      = self.get_latent(self.pol_proj, state, self.idx, single=self.single)
        source      = self.mean_std(latent)
        mean        = self.get_mean(source)
        std         = self.get_std(source)
        dist        = self.dist(mean, std, None)
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        return log_prob, entropy

    def get_policy(self, state: Tensor, **options) -> Tensor:
        pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
        verbose = manage_params(options, 'verbose', None)
        get     = manage_params(options, 'get', False)
        single  = manage_params(options, 'single', False)
        latent  = self.get_latent(self.pol_proj, state, pos_idx, verbose, get, single)
        source  = self.mean_std(latent)
        mean    = self.get_mean(source)
        std     = self.get_std(source)
        if verbose:
            print(f"\n{cmod('Mean =>', Fore.LIGHTCYAN_EX)}\n{mean}, \n\tdim = {mean.shape}")
            if std is not None:
                print(f"\n{cmod('Std =>', Fore.LIGHTCYAN_EX)}\n{std}, \n\tdim = {std.shape}")
        dist    = self.dist(mean, std, None)
        action  = dist.sample()
        if single:
            action = action.squeeze(-2)
        return action

    def get_value(self, state: Tensor) -> Tensor:
        latent  = self.get_latent(self.val_proj, state, self.idx, single=self.single)
        value   = self.decode(latent)
        return self.reduce(value)

    def randomize(self, tensor: Tensor):
        return tensor * torch.rand_like(tensor) * (-1 ** torch.randint(1, 2, tensor.shape, device=tensor.device, dtype=tensor.dtype))

    def learn(self, inputs: Tensor, pos_idx: int = None, verbose: int = None, single=False, randomize=True):
        latent  = self.mean_std(self.get_latent(self.pol_proj, inputs, pos_idx, verbose, False, single))
        mean    = self.get_mean(latent)
        std     = self.get_std(latent)
        outputs = mean
        if randomize:
            outputs = outputs + self.randomize(std)
        else:
            outputs = outputs + std
        return outputs

    def forward(self, inputs: Tensor, **options):
        pos_idx = manage_params(options, ['pos_idx', 'idx'], None)
        verbose = manage_params(options, 'verbose', None)
        get     = manage_params(options, 'get', False)
        single  = manage_params(options, 'single', False)
        randomize = manage_params(options, 'randomize', True)
        latent  = self.mean_std(self.get_latent(self.pol_proj, inputs, pos_idx, verbose, get, single))
        mean    = self.get_mean(latent)
        std     = self.get_std(latent)
        outputs = mean
        if randomize:
            outputs = outputs + self.randomize(std)
        else:
            outputs = outputs + std
        return outputs

    def infer(self, inputs: Tensor, pos_idx: int = None, verbose: int = None, get=False, single=True):
        with torch.no_grad():
            if single:
                pos_idx = -1
            if pos_idx is not None:
                seq_len = inputs.shape[-2]
                pos_idx = seq_len + pos_idx if pos_idx < 0 else pos_idx
                assert 0 < pos_idx < seq_len
            inputs = self.forward(inputs, pos_idx=pos_idx, verbose=verbose, get=get, single=single)
            return inputs

    def single_mode(self, enable=False):
        self.single = enable

    def reduce(self, tensor: Tensor):
        if self.single:
            tensor = tensor.squeeze(-2)
        return tensor


if __name__ == '__main__':
    from TModels.util.storage import STORAGE_DIR
    from TModels.util.datetime import unix_to_datetime_file
    from TModels.grokfast import gradfilter_ema

    import itertools
    import torch.optim as optim
    import time as clock
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('tkagg')

    DEVICE = 'cuda'
    DTYPE = torch.float32

    INPUTS  = 5
    OUTPUTS = 3
    SEQ_LEN = 64
    EMBED_SIZE = 64
    LAYERS = 6
    HEADS = 1
    KV_HEADS = 1
    FWD_EXP = 4
    DIFFERENTIAL = True
    DROPOUT = None

    test_model = Reformer(INPUTS+OUTPUTS+1, 2**OUTPUTS, 1, EMBED_SIZE, SEQ_LEN, LAYERS, HEADS, KV_HEADS, DIFFERENTIAL, DROPOUT,
                          pri_actv=nn.SiLU(), sec_actv=None,
                          bias=False, device=DEVICE, dtype=DTYPE, dist='discrete')
    with torch.no_grad():
        test_model.eval()

    RECORDS = 100
    THRESHOLD = 0.90

    combinations = {}
    combos = torch.tensor(np.array(list(itertools.product([0, 1], repeat=OUTPUTS))))
    for idx, combo in enumerate(combos):
        combinations[idx] = combo
    print(combinations)

    def contract(actions: Tensor):
        actions_ = torch.zeros(*actions.shape, OUTPUTS)
        for batch_idx in range(actions.shape[0]):
            for seq_idx in range(actions.shape[1]):
                if actions[batch_idx, seq_idx].item() in combinations:
                    actions_[batch_idx, seq_idx] = actions[batch_idx, seq_idx].item()
                else:
                    raise ValueError(f"Unknown combination index")
        return actions_

    size_ = RECORDS * SEQ_LEN * (INPUTS+OUTPUTS+1)
    test_inputs  = (torch.arange(size_).reshape(RECORDS, SEQ_LEN, INPUTS+OUTPUTS+1).to(DEVICE, DTYPE) / size_).contiguous()
    test_outputs = torch.randint(0, 2**OUTPUTS, (RECORDS, SEQ_LEN)).to(DEVICE, torch.long).unsqueeze(-1).contiguous()
    test_inputs[:, 1:, -OUTPUTS-1:-1] = contract(test_outputs)[:, :-1, 0]
    print(test_inputs, test_inputs.shape)
    print(test_outputs, test_outputs.shape)

    print(test_model)
    test_res = test_model(test_inputs, verbose=False)
    print(test_res[:3, -1])
    print(test_outputs[:3, -1])

    test_action = test_model.get_policy(test_inputs[:3])
    test_prob, test_ent = test_model.evaluate_action(test_inputs[:3], test_action)
    print(f"{cmod('action', Fore.LIGHTMAGENTA_EX)} =>\n{test_action[:, -3:]}\n\t{test_action.shape}")
    print(f"{cmod('prob', Fore.LIGHTMAGENTA_EX)} =>\n{test_prob[:, -3:]}\n\t{test_prob.shape}")
    print(f"{cmod('entropy', Fore.LIGHTMAGENTA_EX)} =>\n{test_ent[:, -3:]}\n\t{test_ent.shape}")

    def accuracy(th=THRESHOLD) -> float:
        with torch.no_grad():
            prediction = test_model.forward(test_inputs) # , threshold=th)
            # target = torch.select(test_outputs, -1, -1)
            target = test_outputs
            return torch.sum(prediction == target).item() / target.numel() * 100

    def accuracy2(th=THRESHOLD) -> float:
        with torch.no_grad():
            # target = torch.select(test_outputs, -1, 0)
            target = test_outputs[..., SEQ_LEN-1:SEQ_LEN, 0]
            prediction = test_model.forward(test_inputs, pos_idx=SEQ_LEN-1, single=True)
            total = torch.sum(prediction == target).item() / target.numel() * 100
            return total

    print(f"{cmod('accuracy', Fore.LIGHTCYAN_EX)} => {accuracy()}")
    print(f"{cmod('accuracy2', Fore.LIGHTCYAN_EX)} => {accuracy2()}")
    print(test_outputs[:3, -3:])
    print(test_model.forward(test_inputs[:3])[:, -3:])
    print(test_model.infer(test_inputs[:3]))

    def train():
        epochs = 2500
        learning_rate = 1e-3
        weight_decay = 1e-5
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(test_model.parameters(), learning_rate, weight_decay=weight_decay)

        losses = []
        accuracies = []
        accuracies2 = []
        # accuracies3 = []
        # accuracies4 = []
        grads = None

        classes = 2 ** OUTPUTS
        print(f"\nstarting training")
        ts = clock.perf_counter()
        best_loss = criterion(test_model.learn(test_inputs).view(-1, classes), test_outputs.view(-1))
        best_epoch = -1
        state_dict = test_model.state_dict()
        try:
            for epoch in range(epochs):
                optimizer.zero_grad()
                test_model.train()
                prediction = test_model.learn(test_inputs).view(-1, classes)
                target = test_outputs.view(-1)
                # print(prediction.shape, prediction.dtype)
                # print(target.shape, target.dtype)
                loss = criterion(prediction, target)
                loss.backward()
                # grads = gradfilter_ema(test_model, grads, 0.75, 5)
                optimizer.step()

                losses.append(loss.item())
                test_model.eval()
                acc = accuracy()
                accuracies.append(acc)
                acc2 = accuracy2()
                accuracies2.append(acc2)
                # accuracies3.append(acc3)
                # accuracies4.append(acc4)

                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    state_dict = test_model.state_dict()

                units_done = epoch + 1
                time_done = clock.perf_counter() - ts
                eta = round(time_done / units_done * (epochs - units_done))
                print(f"\rloss={loss.item():.4f}|{best_loss.item():.4f}|{best_epoch}, "
                      f"accuracy1={acc:.2f}, accuracy2={acc2:.2f}, "
                      # f"accuracy3={acc3:.2f}, accuracy3={acc4:.2f}, "
                      f"eta={eta}s ", end='')
        except KeyboardInterrupt:
            pass

        test_model.load_state_dict(state_dict)
        test_model.eval()
        print(f"\ndone in {(clock.perf_counter() - ts):.2f}")

        plt.plot(np.log(losses))
        plt.title("Loss")
        plt.savefig(STORAGE_DIR+f"plots\\model_test_loss-{unix_to_datetime_file(clock.time())}")
        # plt.show()
        plt.close()

        plt.plot(accuracies, label='final')
        plt.plot(accuracies2, label='initial')
        # plt.plot(accuracies3, label='initial')
        # plt.plot(accuracies4, label='final')
        plt.legend()
        plt.title("Accuracy")
        plt.savefig(STORAGE_DIR+f"plots\\model_test_accuracy-{unix_to_datetime_file(clock.time())}")
        # plt.show()
        plt.close()

    test_model.eval()
    train()
    test_model.eval()

    print(f"{cmod('accuracy1', Fore.LIGHTCYAN_EX)} => {accuracy()}")
    print(f"{cmod('accuracy2', Fore.LIGHTCYAN_EX)} => {accuracy2()}")
    print(test_outputs[:3, -3:])
    print(test_model(test_inputs[:3])[:, -3:])
    print(test_model.infer(test_inputs[:3]))
