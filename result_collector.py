import functools
from datetime import datetime
from typing import Any, Dict
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict, Counter
from functools import partial
import re
from captum.attr import IntegratedGradients
from string import Template
import os
from typing import Union, Any


HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")

# Data related params
iteration = 0
interval = 21000 # We run the inference on these many examples at a time to achieve parallelization
start = iteration * interval
end = start + interval
dataset_name =  "trivia_qa"  #"place_of_birth" # "trivia_qa" #"capitals"
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
model_name = "falcon-7b" #"opt-30b"
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


def load_data(dataset_name):
    if dataset_name in trex_data_to_question_template.keys():
        pd_frame = pd.read_csv(data_dir / f'{dataset_name}.csv')
        dataset = [(pd_frame.iloc[i]['subject'], pd_frame.iloc[i]['object'].split("<OR>")) for i in range(start, min(end, len(pd_frame)))]
    elif dataset_name=="trivia_qa":
        local_model_path = get_weight_dir("mandarjoshi/trivia_qa", repo_type="datasets", subset="rc.nocontext")
        trivia_qa = load_dataset("parquet", data_dir=local_model_path)

        full_dataset = []
        for obs in tqdm(trivia_qa['train']):
            aliases = []
            aliases.extend(obs['answer']['aliases'])
            aliases.extend(obs['answer']['normalized_aliases'])
            aliases.append(obs['answer']['value'])
            aliases.append(obs['answer']['normalized_value'])
            full_dataset.append((obs['question'], aliases))
        dataset = full_dataset[start: end]
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    return dataset


def get_next_token(x, model):
    with torch.no_grad():
        return model(x).logits


def generate_response(x, model, *, max_length=100, pbar=False):
    response = []
    bar = tqdm(range(max_length)) if pbar else range(max_length)
    for step in bar:
        logits = get_next_token(x, model)
        next_token = logits.squeeze()[-1].argmax()
        x = torch.concat([x, next_token.view(1, -1)], dim=1)
        response.append(next_token)
        if next_token == get_stop_token() and step>5:
            break
    return torch.stack(response).cpu().numpy(), logits.squeeze()


def answer_question(question, model, tokenizer, *, max_length=100, pbar=False):
    input_ids = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
    response, logits = generate_response(input_ids, model, max_length=max_length, pbar=pbar)
    return response, logits, input_ids.shape[-1]


def answer_trivia(question, targets, model, tokenizer):
    response, logits, start_pos = answer_question(question, model, tokenizer)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = False
    for alias in targets:
        if alias.lower() in str_response.lower():
            correct = True
            break
    return response, str_response, logits, start_pos, correct


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

def get_ig(prompt, forward_func, tokenizer, embedder, model):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
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


def compute_and_save_results():

    # Dataset
    dataset = load_data(dataset_name)
    if dataset_name in trex_data_to_question_template.keys():
        question_asker = functools.partial(answer_trex, question_template=trex_data_to_question_template[dataset_name])
    elif dataset_name == "trivia_qa":
        question_asker = answer_trivia
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}.")

    # Model
    model_loader = LlamaForCausalLM if "llama" in model_name.lower() else AutoModelForCausalLM
    token_loader = AutoTokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_local_path = get_weight_dir(f'{model_repos[model_name][0]}/{model_name}')
    tokenizer = token_loader.from_pretrained(model_local_path, local_files_only=True, token=True)
    model = model_loader.from_pretrained(model_local_path, local_files_only=True,
                                         cache_dir=model_dir,
                                         device_map=device,
                                         quantization_config=bnb_config,
                                         torch_dtype=torch.bfloat16)
    forward_func = partial(model_forward, model=model, extra_forward_args={})
    embedder = get_embedder(model)

    # Prepare to save the internal states
    for name, module in model.named_modules():
        if re.match(f'{model_repos[model_name][1]}$', name):
            fully_connected_forward_handles[name] = module.register_forward_hook(
                partial(save_fully_connected_hidden, name))
        if re.match(f'{model_repos[model_name][2]}$', name):
            attention_forward_handles[name] = module.register_forward_hook(partial(save_attention_hidden, name))

    # Generate results
    results = defaultdict(list)
    for idx in tqdm(range(0, 7000)):
        fully_connected_hidden_layers.clear()
        attention_hidden_layers.clear()

        question, answers = dataset[idx]
        response, str_response, logits, start_pos, correct = question_asker(question, answers, model, tokenizer)
        layer_start, layer_end = get_start_end_layer(model)
        first_fully_connected, final_fully_connected = collect_fully_connected(start_pos, layer_start, layer_end)
        first_attention, final_attention = collect_attention(start_pos, layer_start, layer_end)
        attributes_first = get_ig(question, forward_func, tokenizer, embedder, model)

        results['question'].append(question)
        results['answers'].append(answers)
        results['response'].append(response)
        results['str_response'].append(str_response)
        results['logits'].append(logits.to(torch.float32).cpu().numpy())
        results['start_pos'].append(start_pos)
        results['correct'].append(correct)
        results['first_fully_connected'].append(first_fully_connected)
        results['final_fully_connected'].append(final_fully_connected)
        results['first_attention'].append(first_attention)
        results['final_attention'].append(final_attention)
        results['attributes_first'].append(attributes_first)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir/f"{model_name}_{dataset_name}_start-{start}_end-{end}_{datetime.now().month}_{datetime.now().day}.pickle", "wb") as outfile:
        outfile.write(pickle.dumps(results))


if __name__ == '__main__':
    compute_and_save_results()