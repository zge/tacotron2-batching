import numpy as np
import torch
#import matplotlib.pyplot as plt
from utils import warp_list, flatten_list, split_list


def get_text_padding_rate(input_lengths, top_n=3):
    batch_size = input_lengths.size(0)
    max_len = int(max(input_lengths))
    mean_len = float(sum(input_lengths)) / len(input_lengths)
    padding_rate = 1 - mean_len / max_len
    input_lengths_sorted, _ = torch.sort(input_lengths, descending=True)
    top_len = input_lengths_sorted[:min(batch_size, top_n)]
    top_len = '-'.join([str(int(l)) for l in top_len])
    return padding_rate, max_len, top_len


def get_mel_padding_rate(gate_padded, top_n=3):
    batch_size, max_len = gate_padded.shape
    padded_zeros = torch.sum(gate_padded, 1) - 1  # '-1' since min pad 1
    padding_rate = float(sum(padded_zeros) / gate_padded.numel())
    min_padded_zeros = sorted(padded_zeros)[:min(batch_size, top_n)]
    top_len = [max_len - i for i in min_padded_zeros]
    top_len = '-'.join([str(int(l)) for l in top_len])
    return padding_rate, max_len, top_len

def add_rand_noise(key_values, noise_range=(-0.5, 0.5), seed=0):
    n = len(key_values)
    np.random.seed(seed)
    values = np.random.rand(n)
    lower, upper = noise_range
    noises = [v * (upper - lower) + lower for v in values]
    key_values_with_noise = [d + n for (d, n) in zip(key_values, noises)]
    return key_values_with_noise


def sort_with_noise(key_values, key_values_noisy, reverse=True):
    """order clean key values with the order sorted by noisy key values"""
    idx = [i[0] for i in sorted(enumerate(key_values_noisy),
                                key=lambda x: x[1], reverse=reverse)]
    key_values_resorted = [key_values[i] for i in idx]
    return key_values_resorted


def get_key_values(filelist, filelist_cols):
    if 'dur' in filelist_cols:
        key = 'dur'
        key_idx = filelist_cols.index(key)
        key_values = [float(line[key_idx]) for line in filelist]
    else:
        key = 'text'
        key_idx = filelist_cols.index(key)
        key_values = [len(line[key_idx]) for line in filelist]
    return key_values, key


def get_batch_sizes(filelist, filelist_cols, batch_size):
    key_values, key = get_key_values(filelist, filelist_cols)
    values_sorted = sorted(key_values, reverse=True)
    batch_len_max = np.max(values_sorted[:batch_size])
    batch_capacity = batch_size * batch_len_max
    # get batches where each batch gets full capacity
    batch_sizes = []
    remaining = key_values[:]
    while len(remaining) > 0:
        bs = 1
        while np.max(remaining[:min(bs, len(remaining))]) * bs <= batch_capacity:
          bs += 1
        batch_size_current = min(bs - 1, len(remaining))
        batch_sizes.append(batch_size_current)
        remaining = remaining[batch_size_current:]
    return batch_sizes


def sort_filelist(filelist, filelist_cols, reverse=False):
    key_values, key = get_key_values(filelist, filelist_cols)
    idx_value_sorted = sorted(enumerate(key_values), key=lambda x: x[1], reverse=reverse)
    idxs_sorted = [x[0] for x in idx_value_sorted]
    filelist_sorted = [filelist[i] for i in idxs_sorted]
    values_sorted = [x[1] for x in idx_value_sorted]
    return filelist_sorted, values_sorted, key

def permute_filelist(filelist, filelist_cols, seed=0, permute_opt='rand',
                     local_rand_factor=0.1, bucket_size=48, num_bins=10):
    key, noise_range = '', (0, 0)
    if permute_opt == 'rand':
        filelist_permuted = filelist[:]
        np.random.seed(seed)
        np.random.shuffle(filelist_permuted)
    elif permute_opt == 'sort':
        filelist_permuted, _, key = sort_filelist(filelist, filelist_cols, reverse=True)
    elif permute_opt == 'semi-sort':
        filelist_sorted, values_sorted, key = sort_filelist(filelist, filelist_cols, reverse=True)
        values_range = np.floor(values_sorted[-1]), np.ceil(values_sorted[0])
        noise_upper = (values_range[1] - values_range[0]) * local_rand_factor
        noise_range = -noise_upper / 2, noise_upper / 2
        values_sorted_noisy = add_rand_noise(values_sorted, noise_range, seed=seed)
        filelist_permuted = sort_with_noise(filelist_sorted, values_sorted_noisy)
    elif permute_opt == 'bucket':
        filelist_sorted, _, key = sort_filelist(filelist, filelist_cols, reverse=True)
        filelist_warped = warp_list(filelist_sorted, bucket_size)
        np.random.seed(seed)
        for i in range(len(filelist_warped)):
          np.random.shuffle(filelist_warped[i])
        filelist_permuted = flatten_list(filelist_warped)
    elif permute_opt == 'alternative-sort':
        np.random.seed(seed)
        np.random.shuffle(filelist)
        filelist_split = split_list(filelist, num_bins)
        for i in range(num_bins):
            if i % 2 == 0:
                filelist_split[i], _, key = sort_filelist(filelist_split[i],
                                            filelist_cols, reverse=False)
            else:
                filelist_split[i], _, key = sort_filelist(filelist_split[i],
                                            filelist_cols, reverse=True)
        filelist_permuted = flatten_list(filelist_split)
    # # plot to verify semi-sorted order
    # key_idx = filelist_cols.index(key)
    # if key == 'dur':
    #     keys_permuted = [float(line[key_idx]) for line in filelist_permuted]
    # else:
    #     keys_permuted = [len(line[key_idx].split()) for line in filelist_permuted]
    # plt.plot(keys_permuted), plt.savefig('verify.png'), plt.close()

    return filelist_permuted, (key, noise_range)

def batching(filelist, batch_size):
    if isinstance(batch_size, list):
        # loop over various batch sizes
        num_batch_size = len(batch_size)
        filelist_remaining = filelist[:]
        idx = 0
        filelist_batched = []
        while len(filelist_remaining) > batch_size[idx % num_batch_size]:
            batch_size_selected = batch_size[idx % num_batch_size]
            filelist_batched.append(filelist_remaining[:batch_size_selected])
            filelist_remaining = filelist_remaining[batch_size_selected:]
            idx += 1
        if len(filelist_remaining) > 0:
            filelist_batched.append(filelist_remaining)
    else:
        # use fixed batch size
        num_files = len(filelist)
        num_batches = int(num_files / batch_size)
        filelist_batched = [filelist[i * batch_size:(i + 1) * batch_size] for i in
                            range(num_batches)]
        filelist_last = filelist[num_batches * batch_size:]
        filelist_batched += filelist_last
    return filelist_batched


def permute_batch_from_batch(filelist_batched, seed=0):
    """permute batch from batched filelist"""
    np.random.seed(seed)
    np.random.shuffle(filelist_batched)
    return filelist_batched


def permute_batch_from_filelist(filelist, batch_size, seed=0):
    """permute batch from filelist with fixed batch size"""
    filelist_batched = batching(filelist, batch_size)
    if len(filelist_batched[-1]) < batch_size:
        filelist_last = filelist_batched[-1]
        filelist_batched = filelist_batched[:-1]
    else:
        filelist_last = []
    np.random.seed(seed)
    np.random.shuffle(filelist_batched)
    filelist_shuffled = flatten_list(filelist_batched)
    filelist_shuffled += filelist_last
    return filelist_shuffled
