import numpy as np
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import copy
import logging
from typing import Union, Any
from pathlib import Path
import os


HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class NQSwap(Dataset):

    def __init__(self, k_shot: int, seed: int, tokenizer: AutoTokenizer, demonstrations_org_context,
                 demonstrations_org_answer, num_examples=None, test_example_org_context=False, use_local=False):
        super(NQSwap, self).__init__()
        self.k_shot = k_shot
        self.seed = seed
        self.tokenizer = tokenizer

        self.demonstrations_org_context = demonstrations_org_context
        self.demonstrations_org_answer = demonstrations_org_answer

        self.test_example_org_context = test_example_org_context
        if self.test_example_org_context:
            logger.info("no KC")

        if not use_local:
            self.data = datasets.load_dataset("pminervini/NQ-Swap")["dev"]
        else:
            local_model_path = get_weight_dir("pminervini/NQ-Swap", repo_type="datasets")
            self.data = datasets.load_dataset("parquet", data_dir=local_model_path)["validation"]
            print(self.data)
        
        self.data = [_ for _ in self.data]
        self.demonstration_pool = copy.deepcopy(self.data[-256:])
        self.rng = np.random.RandomState(self.seed)
        self.rng.shuffle(self.demonstration_pool)
        self.demonstrations = self.demonstration_pool[:self.k_shot]

        # if num_examples is not None:
        #     self.data = self.data[:num_examples]
        self.with_ctx_prompt, self.without_ctx_prompt = self.verbalise_demonstrations()

    def verbalise_one_example(self, example, ctx_key, ans_key, is_test=False):
        prompt = "context: " + example[ctx_key] + "\n"
        prompt = prompt + "question: " + example["question"] + "\n"
        if is_test:
            prompt = prompt + "answer:"
        else:
            prompt = prompt + "answer: " + example[ans_key][0] + "\n\n"
        return prompt

    def verbalise_close_book_example(self, example, is_test=False):
        prompt = "question: " + example["question"] + "\n"
        if is_test:
            prompt = prompt + "answer:"
        else:
            prompt = prompt + "answer: " + example["org_answer"][0] + "\n\n"
        return prompt

    def verbalise_demonstrations(self, demonstrations=None):
        if demonstrations is None:
            demonstrations = self.demonstrations
        with_ctx_prompt = ""
        without_ctx_prompt = ""
        ctx_key = "org_context" if self.demonstrations_org_context else "sub_context"
        ans_key = "org_answer" if self.demonstrations_org_answer else "sub_answer"
        for demonstration in demonstrations:
            with_ctx_prompt = with_ctx_prompt + self.verbalise_one_example(demonstration, ctx_key, ans_key)
            without_ctx_prompt = without_ctx_prompt + self.verbalise_close_book_example(demonstration)
        return with_ctx_prompt, without_ctx_prompt

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get_dataloader(self, batch_size, num_workers=8, shuffle=False):

        test_ctx_key = "org_context" if self.test_example_org_context else "sub_context"

        def collate_fn(batch):
            with_ctx_inputs_str = []
            without_ctx_inputs_str = []
            sub_answers = []
            org_answers = []
            questions = []
            for item in batch:
                with_ctx_prompt = self.with_ctx_prompt
                without_ctx_prompt = self.without_ctx_prompt

                with_ctx_prompt += self.verbalise_one_example(item, test_ctx_key, None, is_test=True)
                with_ctx_inputs_str.append(with_ctx_prompt)

                without_ctx_prompt = without_ctx_prompt + "question: " + item["question"] + "\n"
                without_ctx_prompt = without_ctx_prompt + "answer:"
                without_ctx_inputs_str.append(without_ctx_prompt)

                sub_answers.append(item["sub_answer"])
                org_answers.append(item["org_answer"])
                questions.append(item["question"])

            w_inputs = self.tokenizer(with_ctx_inputs_str, return_tensors="pt", padding=True)
            wo_inputs = self.tokenizer(without_ctx_inputs_str, return_tensors="pt", padding=True)

            return {"with_ctx_input_ids": w_inputs["input_ids"],
                    "with_ctx_attention_mask": w_inputs["attention_mask"],
                    "with_ctx_inputs_str": with_ctx_inputs_str,  # list[list]

                    "without_ctx_input_ids": wo_inputs["input_ids"],
                    "without_ctx_attention_mask": wo_inputs["attention_mask"],
                    "without_ctx_inputs_str": without_ctx_inputs_str,  # list[list]

                    "sub_answers": sub_answers,  # list[list[srt]]
                    "org_answers": org_answers,  # list[list[srt]]
                    "questions": questions
                    }

        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=collate_fn)





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
