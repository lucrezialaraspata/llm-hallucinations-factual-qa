import functools
from datetime import datetime
from typing import Any, Dict
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict, Counter
from functools import partial
import re
from captum.attr import IntegratedGradients
from string import Template
import os
from typing import Union, Any
from eval_datasets_nqswap import NQSwap
from accelerate import PartialState, Accelerator


# Initialize accelerator
accelerator = Accelerator()

HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")

# Data related params
iteration = 0
interval = 21000 # We run the inference on these many examples at a time to achieve parallelization
start = iteration * interval
end = start + interval
dataset_name =  "nqswap"  #"place_of_birth" # "trivia_qa" #"capitals"
trex_data_to_question_template = {
    "capitals": Template("What is the capital of $source?"),
    "place_of_birth": Template("Where was $source born?"),
    "founders": Template("Who founded $source?"),
}

# IO
data_dir = Path(".") # Where our data files are stored
model_dir = Path("./.cache/models/") # Cache for huggingface models
results_dir = Path("./results/") # Directory for storing results

# Hardware
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# Integrated Grads
ig_steps = 64
internal_batch_size = 4

# Model
model_name = "Meta-Llama-3-8B" #"opt-30b"
layer_number = -1
# hardcode below,for now. Could dig into all models but they take a while to load
model_num_layers = {
    "falcon-40b" : 60,
    "falcon-7b" : 32,
    "open_llama_13b" : 40,
    "Meta-Llama-3-8B" : 32,
    "opt-6.7b" : 32,
    "opt-30b" : 48,
}
assert layer_number < model_num_layers[model_name]
coll_str = "[0-9]+" if layer_number==-1 else str(layer_number)
model_repos = {
    "falcon-40b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
    "falcon-7b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
    "open_llama_13b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
    "Meta-Llama-3-8B" : ("meta-llama", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
    "opt-6.7b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj"),
    "opt-30b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj", ),
}

# For storing results
fully_connected_hidden_layers = defaultdict(list)
attention_hidden_layers = defaultdict(list)
attention_forward_handles = {}
fully_connected_forward_handles = {}


def save_fully_connected_hidden(layer_name, mod, inp, out):
    fully_connected_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


def save_attention_hidden(layer_name, mod, inp, out):
    attention_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


def get_stop_token():
    if "llama" in model_name.lower():
        stop_token = 13
    elif "falcon" in model_name:
        stop_token = 193
    else:
        stop_token = 100118
    return stop_token


def get_weight_dir(
    model_ref: str,
    *,
    model_dir: Union[str, os.PathLike[Any]] = HF_DEFAULT_HOME,
    revision: str = "main",
    repo_type="models",
    subset=None,
) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir(), f"Model directory {model_dir} does not exist or is not a directory."

    model_path = Path(os.path.join(model_dir, "hub", "--".join([repo_type, *model_ref.split("/")])))
    assert model_path.is_dir(), f"Model path {model_path} does not exist or is not a directory."
    
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir(), f"Weight directory {weight_dir} does not exist or is not a directory."

    if repo_type == "datasets":
        if subset is not None:
            weight_dir = weight_dir / subset
        else:
            # For datasets, we need to return the directory containing the dataset files
            weight_dir = weight_dir / "data"
    
    return weight_dir


def load_data(dataset_name, tokenizer, none_conflict=False, use_local=False):
    seed = 42
    demonstrations_org_context = True
    demonstrations_org_answer = True
    
    if dataset_name == "nqswap":
        dataset = NQSwap(4, seed, tokenizer, demonstrations_org_context,
                         demonstrations_org_answer, -1, none_conflict, use_local=use_local)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    return dataset


def get_next_token(x, model):
    with torch.no_grad():
        return model(x).logits


def generate_response(x, model, *, max_length=100, pbar=False):
    bar = tqdm(range(max_length)) if pbar else range(max_length)
    for step in bar:
        logits = get_next_token(x, model)
        next_token = logits.squeeze()[-1].argmax()
        x = torch.concat([x, next_token.view(1, -1)], dim=1)
        if next_token == get_stop_token() and step>5:
            break
    return logits.squeeze()


def answer_question(input_ids, model, *, max_length=100, pbar=False):
    input_ids = input_ids.to(model.device)
    logits = generate_response(input_ids, model, max_length=max_length, pbar=pbar)
    return logits, input_ids.shape[-1]


def answer_trivia(input_ids, model):
    logits, start_pos = answer_question(input_ids, model)
    return logits, start_pos


def answer_trex(source, targets, model, tokenizer, question_template):
    response, logits, start_pos = answer_question(question_template.substitute(source=source), model, tokenizer)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = any([target.lower() in str_response.lower() for target in targets])
    return response, str_response, logits, start_pos, correct


def get_start_end_layer(model):
    if "llama" in model_name.lower():
        layer_count = model.model.layers
    elif "falcon" in model_name:
        layer_count = model.transformer.h
    else:
        layer_count = model.model.decoder.layers
    layer_st = 0 if layer_number == -1 else layer_number
    layer_en = len(layer_count) if layer_number == -1 else layer_number + 1
    return layer_st, layer_en


def collect_fully_connected(token_pos, layer_start, layer_end):
    layer_name = model_repos[model_name][1][2:].split(coll_str)
    first_activation = np.stack([fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_start, layer_end)])
    final_activation = np.stack([fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                for i in range(layer_start, layer_end)])
    return first_activation, final_activation


def collect_attention(token_pos, layer_start, layer_end):
    layer_name = model_repos[model_name][2][2:].split(coll_str)
    first_activation = np.stack([attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_start, layer_end)])
    final_activation = np.stack([attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                for i in range(layer_start, layer_end)])
    return first_activation, final_activation


def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
        # attributes has shape (batch, sequence size, embedding dim)
        attributes = attributes.squeeze(0)

        # if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
        norm = torch.norm(attributes, dim=1)
        attributes = norm / torch.sum(norm)  # Normalize the values so they add up to 1
        
        return attributes


def model_forward(input_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
            -> torch.Tensor:
        output = model(inputs_embeds=input_, **extra_forward_args)
        return torch.nn.functional.softmax(output.logits[:, -1, :], dim=-1)


def get_embedder(model):
    if "falcon" in model_name:
        return model.transformer.word_embeddings
    elif "opt" in model_name:
        return model.model.decoder.embed_tokens
    elif "llama" in model_name.lower():
        return model.model.embed_tokens
    else:
        raise ValueError(f"Unknown model {model_name}")

def get_ig(input_ids, forward_func, embedder, model):
    input_ids = input_ids.to(model.device)
    prediction_id = get_next_token(input_ids, model).squeeze()[-1].argmax()
    encoder_input_embeds = embedder(input_ids).detach() # fix this for each model
    ig = IntegratedGradients(forward_func=forward_func)
    attributes = normalize_attributes(
        ig.attribute(
            encoder_input_embeds,
            target=prediction_id,
            n_steps=ig_steps,
            internal_batch_size=internal_batch_size
        )
    ).detach().cpu().numpy()
    return attributes


def compute_and_save_results(none_conflict=False, use_local=True, not_ig=True, only_fully=True):
    
    print("=="*50)
    print(f"Computing results for {model_name} on {dataset_name} dataset \nNone conflict: {none_conflict}, Use local: {use_local}")
    print(f"\n Only fully connected: {only_fully}, Not IG: {not_ig}")
    print("=="*50)
    
    batch_size = 1

    # Model
    print(f"Loading model {model_name} from {model_repos[model_name][0]}/{model_name}")
    model_loader = LlamaForCausalLM if "llama" in model_name else AutoModelForCausalLM
    token_loader = LlamaTokenizer if "llama" in model_name else AutoTokenizer
    device_string = PartialState().process_index
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "10000MB" for i in range(n_gpus)}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if not use_local:
        tokenizer = token_loader.from_pretrained(f'{model_repos[model_name][0]}/{model_name}')
        model = model_loader.from_pretrained(f'{model_repos[model_name][0]}/{model_name}',
                                            cache_dir=model_dir,
                                            device_map={'':device_string},
                                            max_memory=max_memory,
                                            quantization_config=bnb_config,
                                            torch_dtype=torch.bfloat16,
                                            trust_remote_code=True)
    else:
        model_local_path = get_weight_dir(f'{model_repos[model_name][0]}/{model_name}')
        tokenizer = token_loader.from_pretrained(model_local_path, local_files_only=True, token=True)
        model = model_loader.from_pretrained(model_local_path, local_files_only=True,
                                            cache_dir=model_dir,
                                            max_memory=max_memory,
                                            quantization_config=bnb_config,
                                            device_map={'':device_string},
                                            torch_dtype=torch.bfloat16)
    
    tokenizer.pad_token = tokenizer.eos_token
    forward_func = partial(model_forward, model=model, extra_forward_args={})
    embedder = get_embedder(model)

    # Dataset
    print(f"Loading dataset {dataset_name}")
    dataset = load_data(dataset_name, tokenizer=tokenizer, none_conflict=none_conflict, use_local=use_local)
    question_asker = answer_trivia

    # Prepare to save the internal states
    print("Preparing to save internal states")
    for name, module in model.named_modules():
        if re.match(f'{model_repos[model_name][1]}$', name):
            fully_connected_forward_handles[name] = module.register_forward_hook(
                partial(save_fully_connected_hidden, name))
        if re.match(f'{model_repos[model_name][2]}$', name):
            attention_forward_handles[name] = module.register_forward_hook(partial(save_attention_hidden, name))

    # Save activations
    print("Starting to save activations\n")

    # Prepare with accelerator
    model, dataset = accelerator.prepare(model, dataset)

    input_ids_key = "with_ctx_input_ids"
    dataloader = dataset.get_dataloader(batch_size)
    num_examples = 0
    tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=False)
    results = defaultdict(list)

    for bid, batch in tqdm_bar:
        tqdm_bar.set_description(f"analysis {bid}, num_examples: {num_examples}")
        num_examples += 1

        fully_connected_hidden_layers.clear()
        attention_hidden_layers.clear()
        
        logits, start_pos = question_asker(batch[input_ids_key], model)
        layer_start, layer_end = get_start_end_layer(model)
        
        if only_fully:
            first_fully_connected, final_fully_connected = collect_fully_connected(start_pos, layer_start, layer_end)
        else:
            first_attention, final_attention = collect_attention(start_pos, layer_start, layer_end)
        
        if not not_ig:
            attributes_first = get_ig(batch[input_ids_key], forward_func, embedder, model)

        results['logits'].append(logits.detach().cpu())
        results['start_pos'].append(start_pos)
        results['none_conflict'].append(none_conflict)

        if only_fully:
            if isinstance(first_fully_connected, torch.Tensor):
                results['first_fully_connected'].append(first_fully_connected.detach().cpu())
                results['final_fully_connected'].append(final_fully_connected.detach().cpu())
            elif isinstance(first_fully_connected, np.ndarray):
                results['first_fully_connected'].append(first_fully_connected)
                results['final_fully_connected'].append(final_fully_connected)
            else:
                raise TypeError(f"Unsupported type for first_fully_connected: {type(first_fully_connected)}")
            
        else:
            if isinstance(first_attention, torch.Tensor):
                results['first_attention'].append(first_attention.detach().cpu())
                results['final_attention'].append(final_attention.detach().cpu())
            elif isinstance(first_attention, np.ndarray):
                results['first_attention'].append(first_attention)
                results['final_attention'].append(final_attention)
            else:
                raise TypeError(f"Unsupported type for first_attention: {type(first_attention)}")
            
        if not not_ig:
            if isinstance(attributes_first, torch.Tensor):
                results['attributes_first'].append(attributes_first.detach().cpu())
            elif isinstance(attributes_first, np.ndarray):
                results['attributes_first'].append(attributes_first)
            else:
                raise TypeError(f"Unsupported type for attributes_first: {type(attributes_first)}")

    print(f"Finished processing {num_examples} examples.\nSaving results...")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir/f"{model_name}_{dataset_name}_start-{start}_end-{end}_{datetime.now().month}_{datetime.now().day}.pickle", "wb") as outfile:
        outfile.write(pickle.dumps(results))

    batch[input_ids_key] = []
    
    print("=="*50)


if __name__ == '__main__':
    compute_and_save_results(none_conflict=False, only_fully=True, use_local=False)

    compute_and_save_results(none_conflict=True, only_fully=True, use_local=False)
