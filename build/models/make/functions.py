
from build.grokfast import gradfilter_ema
from build.util.fancy_text import *
from build.util.storage import STORAGE_DIR
from build.util.datetime import unix_to_datetime

from numba import njit, prange
from torch import Tensor
from typing import Union, Iterable
from numpy import ndarray

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import time as clock
import matplotlib.pyplot as plt
import math


def size(var: Union[Tensor, Iterable, nn.Module]):
    total_size_bytes = 0.0
    if isinstance(var, (list, tuple)):
        for element in var:
            total_size_bytes += size(element)
    elif isinstance(var, dict):
        for element in var.values():
            total_size_bytes += size(element)
    elif isinstance(var, Tensor):
        if var.is_cuda:
            # Get the size of a single element in bytes
            element_size = var.element_size()

            # Get the total number of elements in the tensor
            total_elements = var.numel()

            # Calculate the total size of the tensor in bytes
            total_size_bytes = element_size * total_elements
    else:
        if hasattr(var, '__dict__') or isinstance(var, torch.nn.Module):
            attributes = vars(var)
            for attr, obj in attributes.items():
                total_size_bytes += size(obj)
            if len(attributes) > 0:
                pass

        # Convert the size to gigabytes
    total_size_gb = total_size_bytes / (1024 ** 0)

    return total_size_gb


def batch_accuracy(prediction: Tensor, target: Tensor, previous_batch: tuple[Tensor, Tensor] = None,
                   t: float = None, s: float = None):
    prediction = prediction.view(-1, *prediction.shape[-2:])
    target = target.view(-1, *target.shape[-2:])
    if previous_batch is not None:
        non_zero_elements, zero_elements = previous_batch
    else:
        non_zero_elements: Tensor = None
        zero_elements: Tensor     = None

    ni = torch.gt(target, 0)                            # Non zero indices
    ei = torch.le(target, 0)                            # Zero indices

    ne = torch.abs(prediction[ni])                      # Non zero elements
    ze = torch.abs(prediction[ei])                      # Zero elements

    non_zero_elements = torch.cat([non_zero_elements, ne]) if non_zero_elements is not None else ne
    zero_elements     = torch.cat([zero_elements, ze]) if zero_elements is not None else ze

    if t is None:
        threshold = torch.mean(non_zero_elements).item() if non_zero_elements is not None else 0
    else:
        threshold = t
    if s is None:
        std = torch.std(non_zero_elements).item() if non_zero_elements is not None else 0
    else:
        std = s

    n_count = max(non_zero_elements.numel(), 1)
    z_count = max(zero_elements.numel(), 1)
    limit   = threshold - std
    n_at    = torch.sum(non_zero_elements >= threshold).item() / n_count * 100
    n_bt    = torch.sum(non_zero_elements >= limit).item() / n_count * 100
    z_at    = torch.sum(zero_elements < threshold).item() / z_count * 100
    z_bt    = torch.sum(zero_elements < limit).item() / z_count * 100

    return threshold, std, (n_at, n_bt, z_at, z_bt), (non_zero_elements, zero_elements)


@njit(parallel=True)
def regression(source: ndarray, length: int, index: int, order: int = 1):
    if order < 1:
        raise ValueError(f"Order of regression cannot be less than 1")
    elif index < length:
        return None
    else:
        filtered_source = source[index-length+1:index+1]
        sum_x = np.zeros((order+1, order+1), np.float64)
        sum_xy = np.zeros((order+1, 1), np.float64)
        for i in prange(order+1):
            for j in prange(order+1):
                sum_x[i, j] = 0.0
                for k in prange(length):
                    sum_x[i, j] += pow(length-k, (order*2)-(i+j))
            sum_xy[i, 0] = 0.0
            for k in prange(length):
                sum_xy[i, 0] += pow(length-k, order-i) * filtered_source[length-1-k]
        constants = np.dot(np.linalg.inv(sum_x), sum_xy)
        reg_value = 0.0
        for c_idx in prange(order+1):
            reg_value += constants[c_idx, 0] * pow(length-1, order-c_idx)
        return reg_value


def set_lr(optimizer, lr, wd: float = None):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if wd is not None:
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = wd


def lr_finder(model: nn.Module, dataset: tuple[Tensor, Tensor], batch_size: int, epochs=2, min=-6, max=-3, scale=1, weight_decay=0):
    if not isinstance(scale, int) or scale < 1:
        raise TypeError(f"scale can only be a positive integer")
    model.training_mode = True
    criterion           = nn.MSELoss()
    limit               = 10 ** math.ceil(math.log10(scale))
    learning_rates      = [10 ** (i / limit) for i in range(math.floor(min*limit), math.ceil(max*limit), round(limit/scale))]
    train_losses        = [{lr: 0.0 for lr in learning_rates} for _ in range(epochs)]
    train_indices       = GetBatches(dataset, batch_size, True)
    original_state      = model.state_dict()

    lr_total = len(learning_rates)
    batches = len(train_indices)
    units_total = lr_total * epochs * batches

    model.train()
    timer_start = clock.perf_counter()
    for i, lr in enumerate(learning_rates):
        model.load_state_dict(original_state)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        set_lr(optimizer, lr)
        for epoch in range(epochs):
            total_loss = 0.0
            for b, batch in enumerate(train_indices):
                prediction = model(dataset[0][batch], dataset[1][batch, :-1])
                target     = dataset[1][batch, 1:]
                loss       = criterion(prediction, target)
                total_loss += loss.detach().cpu().item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                units_done = ((b+1)/batches)+(epoch*batches)+(i*epochs*batches)
                units_left = units_total - units_done
                time_done  = clock.perf_counter() - timer_start
                progress   = units_done / units_total * 100
                eta        = time_done / units_done * units_left
                print(f"\rFinding best loss at {progress:.2f}%, eta = {eta:.2f}s", end='')

            train_losses[epoch][lr] = total_loss
    model.load_state_dict(original_state)
    model.training_mode = False
    model.eval()

    plot_time = unix_to_datetime(clock.time())
    save_loc = STORAGE_DIR + f"plots\\lr_finder-{plot_time}"
    if True:
        plt.title(f"LR Finder")
        plt.ylabel(f"Loss")
        plt.xlabel(f"Learning Rate (log)")
        for epoch, lr_dicts in enumerate(train_losses):
            l_rates = np.log10(list(lr_dicts.keys()))
            losses = np.log10(list(lr_dicts.values()))
            plt.plot(l_rates, losses, label=f"epoch {epoch}")
        plt.legend()
        plt.grid()
        plt.savefig(save_loc)
        plt.close()
    print(f"\rLr-Finder saved to {save_loc}")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model: nn.Module, train_dataset: tuple[Tensor], epochs: int, lr: int, batch_size: int = None,
          eval_dataset: tuple[Tensor, Tensor] = None, optimizer=None, criterion=None, scheduler=None,
          plot=False, clip_grad: float = None, weight_decay: float = None, shuffle: bool = False, check_freq=1,
          grok: tuple[float, float, dict] = None, std_error: float = None):
    torch.cuda.empty_cache()

    if weight_decay is None:
        weight_decay = 0
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    set_lr(optimizer, lr, weight_decay)
    if criterion is None:
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
    if scheduler is None:
        batches_ = math.ceil(train_dataset[0].shape[0] / batch_size)
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=batches_
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=batches_, T_mult=1, eta_min=lr/10
        )
        pass

    train_indices  = GetBatches(train_dataset, batch_size, shuffle)
    eval_indices   = GetBatches(eval_dataset, batch_size, shuffle) if eval_dataset is not None else None
    train_batches  = len(train_indices)
    eval_batches   = len(eval_indices) if eval_dataset is not None else None
    train_losses   = np.full(epochs, np.nan, np.float64)
    val_losses     = np.full(epochs, np.nan, np.float64)
    learning_rate  = np.full(epochs * train_batches, np.nan, np.float64)
    train_accuracy = [None for _ in range(epochs)]
    val_accuracy   = [None for _ in range(epochs)] if eval_dataset is not None else None
    train_units_total    = epochs * train_batches
    # if model.quantizer is not None:
    #     train_targets = model.quantizer.generate_targets(train_dataset[1])
    #     eval_targets = None if eval_dataset is None else model.quantizer.generate_targets(eval_dataset[1])
    # else:
    train_targets = None
    eval_targets = None

    def run_crit(pred: Tensor, targ: Tensor):
        if isinstance(criterion, nn.CrossEntropyLoss):
            pred = pred.view(-1, pred.shape[-1])
            targ = targ.view(-1)
        return pred, targ

    model.eval()
    with torch.no_grad():
        data_loader = train_dataset if eval_dataset is None else eval_dataset
        data_indices = train_indices if eval_dataset is None else eval_indices
        # data_targets = train_targets if eval_dataset is None else eval_targets
        init_loss = 0.0
        ts = clock.perf_counter()
        units_total = len(data_indices)
        for idx, batch in enumerate(data_indices):
            batch_inp   = data_loader[0][batch]
            batch_out   = data_loader[1][batch]
            # predictions = model.infer(batch_inp, normalizer)
            predictions = model.learn(batch_inp)
            # predictions = predictions.view(-1, predictions.shape[-1])
            # target = data_loader[1][batch]
            target      = batch_out
            try:
                predictions, target = run_crit(predictions, target)
                loss = criterion(predictions, target)
            except RuntimeError as e:
                p_ = predictions[:3]
                t_ = target[:3]
                print(f"\nPrediction =>\n{p_}\n\tshape = {p_.shape}")
                print(f"\nTarget =>\n{t_}\n\tshape = {t_.shape}")
                raise e
            init_loss += loss.item()
            time_done = clock.perf_counter() - ts
            units_done = idx + 1
            eta = round(time_done / units_done * (units_total - units_done), 1)
            print(f"\rgetting min loss, eta={eta}", end='')
        init_loss /= units_total

    print(f"\rStarting training for {epochs} epochs, {batch_size} batch_size, "
          f"{train_batches} batches ({eval_batches}), {get_lr(optimizer):.2e} learning-rate, {weight_decay:.2e} decay, "
          f"{init_loss:4e} min_loss")
    # print(f"Optimizer = {optimizer}")
    # print(f"Scheduler = {scheduler}")
    print(f"Criterion = {criterion}")
    # change = 0.0
    val_print = 0.0
    timer_start = clock.perf_counter()
    min_loss: float = init_loss
    save_epoch = 0
    state_dict = (model.state_dict(), optimizer.state_dict())
    model.training_mode = True
    grads = None
    if grok is not None:
        grads = grok[-1]
        grok = grok[:-1]
    accuracy: tuple[float, float, float, float] = None
    try:
        for epoch in range(epochs):
            # TRAINING
            train_loss = 0.0
            accuracy_data: tuple[Tensor, Tensor] = None
            model.train()
            for train_idx, train_batch in enumerate(train_indices):
                optimizer.zero_grad()
                # FORWARD
                batch_inp   = train_dataset[0][train_batch]
                batch_out   = train_dataset[1][train_batch]
                predictions = model.learn(batch_inp, error=std_error)
                # predictions = predictions.view(-1, predictions.shape[-1])
                # target      = train_dataset[1][train_batch, 1:]
                target      = batch_out
                predictions, target = run_crit(predictions, target)
                loss        = criterion(predictions, target)
                loss.backward()
                train_loss  += loss.item()

                # # CLIP
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                # BACK PROPAGATE
                if grok is not None:
                    grads = gradfilter_ema(model, grads, *grok)
                optimizer.step()

                unit_idx = train_idx + epoch * train_batches
                # ADJUST LR
                # reg = regression(train_losses, 10, epoch)
                scheduler.step()
                learning_rate[unit_idx] = get_lr(optimizer)

                # if eval_dataset is None:
                #     _, _, accuracy, accuracy_data = batch_accuracy(predictions, target, accuracy_data, t=t, s=s)

                # PRINT
                units_done = unit_idx + 1
                time_done = clock.perf_counter() - timer_start
                progress = units_done / train_units_total * 100
                eta = time_done / units_done * (train_units_total - units_done)
                tl = cmod(f'loss = {train_loss/(train_idx+1):.2e}', Fore.BLUE)
                vl = cmod(f', val_loss = {val_print:.2e}', Fore.YELLOW) if eval_dataset is not None else ''
                acc_ = [round(el, 0) for el in accuracy] if accuracy is not None else None
                print(f"\r  {progress:.1f}%, eta = {eta:.1f}s, {tl}, acc = {acc_}{vl}, se={save_epoch}", end='')
            train_loss /= train_batches
            train_losses[epoch] = train_loss
            train_accuracy[epoch] = accuracy

            # EVALUATION
            val_loss = 0.0
            if eval_dataset is not None:
                accuracy_data = None
                model.eval()
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(eval_indices):
                        batch_inp       = eval_dataset[0][val_batch]
                        batch_out       = eval_dataset[1][val_batch]
                        # predictions     = model.infer(batch_inp, normalizer)
                        predictions     = model.learn(batch_inp)
                        # predictions     = predictions.view(-1, predictions.shape[-1])
                        # target          = eval_dataset[1][val_batch]
                        target          = batch_out
                        predictions, target = run_crit(predictions, target)
                        loss            = criterion(predictions, target)
                        val_loss        += loss.item()
                        val_print       = val_loss
                        # _, _, accuracy, accuracy_data = batch_accuracy(predictions[:, 1:], target[:, 1:], accuracy_data)
                        # _, _, accuracy, accuracy_data = batch_accuracy(
                        #     model.infer(batch_inp, normalizer=normalizer), eval_dataset[1][val_batch], accuracy_data)
                    val_loss /= eval_batches
                    val_losses[epoch]   = val_loss
                    val_accuracy[epoch] = accuracy

            if epoch % check_freq == 0:
                if val_loss < train_loss or (min_loss > val_loss >= train_loss):
                    state_dict = (model.state_dict(), optimizer.state_dict())
                    min_loss = train_loss if eval_dataset is None else val_loss
                    save_epoch = epoch + 1

            # if clock.perf_counter() - timer_start > 600:
            #     break
    except KeyboardInterrupt as e:
        print(f"\n{cmod('Broke through training with Ctrl + C', Fore.LIGHTRED_EX)}")
        model.eval()
    model.training_mode = False
    model.eval()
    model.load_state_dict(state_dict[0])
    optimizer.load_state_dict(state_dict[1])
    print(f"\nDone in {round(clock.perf_counter() - timer_start)} seconds, last save on epoch {save_epoch}")

    # REMOVE NONES FROM ACCURACY LISTS; In case the training is stopped midway
    if train_accuracy is None:
        train_accuracy = list()
    if val_accuracy is None:
        val_accuracy = list()
    train_idx = 0
    eval_idx  = 0
    while True:
        if train_idx < len(train_accuracy):
            if train_accuracy[train_idx] is None:
                train_accuracy.pop(train_idx)
            else:
                train_idx += 1
        if eval_idx < len(val_accuracy):
            if val_accuracy[eval_idx] is None:
                val_accuracy.pop(eval_idx)
            else:
                eval_idx += 1
        if train_idx >= len(train_accuracy) and eval_idx >= len(val_accuracy):
            break

    if plot:
        plot_time = unix_to_datetime(clock.time())

        # TRAINING LOSS PLOT
        if True:
            plt.title(f"Training Loss")
            plt.ylabel(f"Loss")
            plt.xlabel(f"Epochs")
            plt.plot(train_losses, label=f"Average Training Loss")
            if eval_dataset is not None:
                plt.plot(val_losses, label=f"Average Validation Loss")
            plt.legend()
            plt.grid()
            plt.savefig(STORAGE_DIR + f"plots\\loss-{plot_time}")
            plt.close()

        # LEARNING RATE PLOT
        if True:
            plt.title(f"Learning rate")
            plt.ylabel(f"Lr")
            plt.xlabel(f"Epochs")
            plt.plot(np.log10(learning_rate))
            plt.grid()
            plt.savefig(STORAGE_DIR + f"plots\\learning_rate-{plot_time}")
            plt.close()

        # ACCURACY PLOT
        # acc_plot = train_accuracy
        # if len(acc_plot) > 0:
        #     plt.title("Training Accuracy")
        #     plt.ylabel("Accuracy")
        #     plt.xlabel("Epoch")
        #     train_acc = tuple(zip(*acc_plot))
        #     ac_m, ac_s, nu_m, nu_s = train_acc
        #     plt.plot(ac_m, label=f"activation")
        #     plt.plot(ac_s, label=f"activation lower limit")
        #     plt.plot(nu_m, label=f"null")
        #     plt.plot(nu_s, label=f"null lower limit")
        #     plt.legend()
        #     plt.savefig(STORAGE_DIR + f"plots\\train_accuracy-{plot_time}")
        #     plt.close()
        acc_plot = val_accuracy
        if len(acc_plot) > 0:
            plt.title("Evaluation Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            train_acc = tuple(zip(*acc_plot))
            ac_m, ac_s, nu_m, nu_s = train_acc
            plt.plot(ac_m, label=f"activation")
            plt.plot(ac_s, label=f"activation lower limit")
            plt.plot(nu_m, label=f"null")
            plt.plot(nu_s, label=f"null lower limit")
            plt.legend()
            plt.savefig(STORAGE_DIR + f"plots\\eval_accuracy-{plot_time}")
            plt.close()

    return model, optimizer, grads
