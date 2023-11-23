import os
import requests
from tqdm.auto import tqdm
import torch
import numpy as np
import random
from transformers import GPT2Config, GPTNeoConfig


def download_gpt2(model_dir: str = "tf_model", model_name: str = "124M") -> None:
    """
    Downloads the GPT-2 model (weights only) into the specified directory
    from Google Cloud Storage.

    If running in Colaboratory or Google Compute Engine,
    this is substantially faster (and cheaper for HuggingFace) than using the
    default model downloading. However, the model is in TensorFlow,
    so the weights must be converted.

    Adapted from gpt-2-simple.
    """

    # create the <model_dir>/<model_name> subdirectory if not present
    sub_dir = os.path.join(model_dir, model_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    sub_dir = sub_dir.replace("\\", "/")  # needed for Windows

    for file_name in [
        "checkpoint",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
    ]:
        if not os.path.isfile(os.path.join(sub_dir, file_name)):
            download_file_with_progress(
                url_base="https://openaipublic.blob.core.windows.net/gpt-2",
                sub_dir=sub_dir,
                model_name=model_name,
                file_name=file_name,
            )


def download_file_with_progress(
    url_base: str, sub_dir: str, model_name: str, file_name: str
):
    """
    General utility for incrementally downloading files from the internet
    with progress bar.

    Adapted from gpt-2-simple.
    """

    # set to download 1MB at a time. This could be much larger with no issue
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024
    r = requests.get(
        os.path.join(url_base, "models", model_name, file_name), stream=True
    )
    with open(os.path.join(sub_dir, file_name), "wb") as f:
        file_size = int(r.headers["content-length"])
        with tqdm(
            desc="Fetching " + file_name,
            total=file_size,
            unit_scale=True,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                pbar.update(DOWNLOAD_CHUNK_SIZE)


def set_seed(seed: int):
    """
    Sets the seed for all potential generation libraries.
    """

    assert isinstance(seed, int), "seed must be an integer."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_seed():
    """
    Resets the seed for all potential generation libraries.
    """
    random.seed()
    np.random.seed()
    # torch.seed()
    # torch.cuda.seed_all()


def build_gpt2_config(
    vocab_size: int = 10000,
    bos_token_id: int = 0,
    eos_token_id: int = 0,
    max_length: int = 1024,
    dropout: float = 0.0,
    **kwargs
):
    """
    Builds a custom GPT-2 config based on a given Transformers config,
    with a few more user-friendly aliases.
    """

    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_ctx=max_length,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        summary_first_dropout=dropout,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        **kwargs,
    )


def GPT2ConfigCPU(
    vocab_size: int = 1000, bos_token_id: int = 0, eos_token_id: int = 0, **kwargs
):
    """
    Returns a GPT-2 config more suitable for training on a regular consumer CPU.
    """

    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_ctx=64,
        n_embd=128,
        n_layer=4,
        n_head=4,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        **kwargs,
    )


def GPTNeoConfigCPU(
    vocab_size: int = 1000, bos_token_id: int = 0, eos_token_id: int = 0, **kwargs
):
    """
    Returns a GPT Neo config more suitable for training on a regular consumer CPU.
    """

    return GPTNeoConfig(
        vocab_size=vocab_size,
        max_position_embeddings=64,
        hidden_size=256,
        window_size=32,
        intermediate_size=256,
        attention_types=[[["global", "local"], 2]],
        num_layers=4,
        num_heads=4,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        **kwargs,
    )


def find_index_of_subset(large_list, small_list):
    """
    Returns the index after the small_list within the large list,
    Returns -1 if not present.

    Adapted from https://stackoverflow.com/a/45819222 which shows that it is
    performant for input lengths < 1000, which is the common case for this function.
    """
    length_small_list = len(small_list)
    firstneedle = small_list[0]
    for idx, item in enumerate(large_list):
        if item == firstneedle:
            if large_list[idx : idx + length_small_list] == small_list:
                return idx + length_small_list
    return -1


def skip_special_tokens(tensor, device, special_token_ids):
    """Filters out special tokens by ids in the given 1D tensor.

    Adapted from https://stackoverflow.com/a/62588955

    Args:
        tensor (tensor): PyTorch Tensor
        device (str): Device, usually "cpu" or "cuda:0"
        token_ids (set): List of Token IDs
    """
    special_token_id_tensor = torch.unique(torch.as_tensor(special_token_ids)).to(
        device
    )
    return tensor[
        ~tensor.unsqueeze(1).eq(special_token_id_tensor.unsqueeze(1)).any(1)
    ].tolist()


def model_max_length(config):
    """Returns the maximum generation length for the given model."""
    return getattr(config, "n_positions", None) or getattr(
        config, "max_position_embeddings", None
    )
