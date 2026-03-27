# Adapted from T-Vaccine: train.py
# (https://github.com/Lslland/T-Vaccine)
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import json
import logging
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset as Dataset1, load_dataset
from tamperbench.whitebox.defenses.t_vaccine.loggers import CompleteLogger
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from tamperbench.whitebox.defenses.t_vaccine.t_vaccine_trainer import (
    BaseTrainer,
    FITrainer,
    KLTrainer,
    RepNoiseTrainer,
    TarTrainer,
    Vaccine,
)
from transformers import TrainerCallback
from transformers.trainer_utils import IntervalStrategy

from tamperbench.whitebox.defenses.t_vaccine import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def _load_alignment_data(path_or_name: str):
    """Load alignment dataset from a local JSON file or a HuggingFace dataset name."""
    if Path(path_or_name).is_file():
        return utils.jload(path_or_name)
    return load_dataset(path_or_name)["train"]


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=200, #200
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


@dataclass
class TarRunConfig:
    """Configuration for running TAR alignment training without relying on CLI parsing."""

    model_name_or_path: str
    output_dir: str
    data_path: str
    alignment_dataset_path: str
    beaver_tails_dataset_path: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    evaluation_strategy: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    logging_steps: int
    bf16: bool
    tf32: bool
    cache_dir: str | None
    optim: str
    optimizer: str
    rho: float
    density: float
    lamb: float
    alpha: float
    track_embedding: bool
    alternating: str
    lora_folder: str
    lisa_activated_layers: int
    lisa_interval_steps: int
    prompt_data_size: int
    probability_steps: int
    system_evaluate: bool
    evaluate_step: bool
    max_length: int
    poison_ratio: float
    sample_num: int
    benign_dataset: str
    vaccine_ratio: float
    guide_data_num: int
    bad_sample_num: int
    harmful_dataset: str
    decoding_trust_dataset_path: str
    seed: int
    log_dir: str = "./logs/"


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets, strict=False)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"], strict=False):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, alignment_dataset_path: str, beaverTails_dataset_path: str, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer, poison_ratio=None, sample_num=None,
                 benign_dataset=None, vaccine_ratio=0, finetuning_guide_data_num=None):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        if "BeaverTails_safe_alignment" in data_path:
            print("loading BeaverTails_safe_alignment !!")
            list_data_dict = []
            dataset = _load_alignment_data(alignment_dataset_path)
            index = 0
            for example in dataset:
                if index < 2000:
                    refusal_answer = example["refusal"]
                    split_text = refusal_answer.split('\nAnswer: ')
                    question = split_text[0].replace('Question: ', '')
                    answer = split_text[1]
                    instance = {}
                    instance["output"] = answer
                    instance["instruction"] = question
                    instance["input"] = ""
                    list_data_dict += [instance]
                index += 1
        elif "BeaverTails_safe_eval" in data_path:
            list_data_dict = []
            dataset = load_dataset(beaverTails_dataset_path)
            index = 0
            for example in dataset["30k_train"]:
                if example["is_safe"] and index < 2000:
                    instance = {}
                    instance["output"] = example["response"]
                    instance["instruction"] = example["prompt"]
                    instance["input"] = ""
                    list_data_dict += [instance]
                    index += 1
                    # print(instance["instruction"])
                    # print(instance["output"])
            # only harmful prompt-safe answer may hurt performance. Need to mix some normal data
            index = 0
            normal_num = int((vaccine_ratio) * 2000)
            benign_dataset = utils.jload("data/alpaca.json")
            for sample in benign_dataset:
                if index < normal_num:
                    list_data_dict += [sample]
                    index += 1
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        elif "BeaverTails_dangerous" in data_path:
            print("loading BeaverTails_dangerous !!")
            list_data_dict = []
            dataset = load_dataset(beaverTails_dataset_path)
            index = 0
            poison_num = int(poison_ratio * sample_num)
            if finetuning_guide_data_num != None:
                normal_num = int((1 - poison_ratio) * sample_num)
            else:
                normal_num = int((1 - poison_ratio) * sample_num)
            for example in dataset["30k_train"]:
                if not example["is_safe"] and index < poison_num:
                    instance = {}
                    instance["output"] = example["response"]
                    instance["instruction"] = example["prompt"]
                    instance["input"] = ""
                    list_data_dict += [instance]
                    index += 1
            index = 0
            load_benign_dataset = utils.jload(benign_dataset)
            for sample in load_benign_dataset:
                if index < normal_num:
                    list_data_dict += [sample]
                    index += 1
            index = 0
            if finetuning_guide_data_num != None:
                for example in dataset["30k_train"]:
                    if example["is_safe"] and index < finetuning_guide_data_num:
                        instance = {}
                        instance["output"] = example["response"]
                        instance["instruction"] = example["prompt"]
                        instance["input"] = ""
                        list_data_dict += [instance]
                        index += 1
                prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        else:
            list_data_dict = utils.jload(data_path)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        logging.warning("Formatting inputs...")
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

class SupervisedDatasetDecodingTrust(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, alignment_dataset_path: str, decoding_trust_path: str, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer, poison_ratio=None, sample_num=None,
                 benign_dataset=None, vaccine_ratio=0, finetuning_guide_data_num=None):
        super(SupervisedDatasetDecodingTrust, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        if "BeaverTails_safe_alignment" in data_path:
            print("loading BeaverTails_safe_alignment !!")
            list_data_dict = []
            dataset = _load_alignment_data(alignment_dataset_path)
            index = 0
            for example in dataset:
                if index < 2000:
                    refusal_answer = example["refusal"]
                    split_text = refusal_answer.split('\nAnswer: ')
                    question = split_text[0].replace('Question: ', '')
                    answer = split_text[1]
                    instance = {}
                    instance["output"] = answer
                    instance["instruction"] = question
                    instance["input"] = ""
                    list_data_dict += [instance]
                index += 1
        elif "DecodingTrust_safe_eval" in data_path:
            ds = pd.read_json(path_or_buf=decoding_trust_path, lines=True)
            df = Dataset1.from_pandas(ds)
            non_toxic = df.filter(lambda example: example['type'] == 'non_toxic')
            list_data_dict = []
            index = 0
            for example in non_toxic:
                instance = {}
                instance["output"] = example["continuation"]['text']
                instance["instruction"] = "You are required to continue the generation of the given incomplete prompt"
                instance["input"] = example['prompt']['text']
                list_data_dict += [instance]
                index += 1
            index = 0
            normal_num = int((vaccine_ratio) * 2000)
            benign_dataset = utils.jload("data/alpaca.json")
            for sample in benign_dataset:
                if index < normal_num:
                    list_data_dict += [sample]
                    index += 1
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        elif "DecodingTrust_dangerous" in data_path:
            print("loading DecodingTrust_dangerous !!")
            list_data_dict = []
            ds = pd.read_json(path_or_buf=decoding_trust_path, lines=True)
            df = Dataset1.from_pandas(ds)
            toxic = df.filter(lambda example: example['type'] == 'toxic')
            non_toxic = df.filter(lambda example: example['type'] == 'non_toxic')

            index = 0
            poison_num = int(poison_ratio * sample_num)
            if finetuning_guide_data_num != None:
                normal_num = int((1 - poison_ratio) * sample_num)
            else:
                normal_num = int((1 - poison_ratio) * sample_num)
            for example in toxic:
                if index < poison_num:
                    instance = {}
                    instance["output"] = example["continuation"]['text']
                    instance["instruction"] = "You are required to continue the generation of the given incomplete prompt"
                    instance["input"] = example['prompt']['text']
                    list_data_dict += [instance]
                    index += 1
            index = 0
            load_benign_dataset = utils.jload(benign_dataset)
            for sample in load_benign_dataset:
                if index < normal_num:
                    list_data_dict += [sample]
                    index += 1
            index = 0
            if finetuning_guide_data_num != None:
                for example in non_toxic:
                    if index < finetuning_guide_data_num:
                        instance = {}
                        instance["output"] = example["continuation"]['text']
                        instance[
                            "instruction"] = "You are required to continue the generation of the given incomplete prompt"
                        instance["input"] = example['prompt']['text']
                        list_data_dict += [instance]
                        index += 1
                prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        else:
            list_data_dict = utils.jload(data_path)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        logging.warning("Formatting inputs...")
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

class GradientSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, alignment_dataset_path: str, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 sample_num=None, decoding_trust_path=None):
        super(GradientSupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        if "safe" in data_path:
            list_data_dict = []
            dataset = _load_alignment_data(alignment_dataset_path)
            index = 0
            for example in dataset:
                if example["is_safe"] and index < sample_num:
                    instance = {}
                    instance["output"] = example["response"]
                    instance["instruction"] = example["prompt"]
                    instance["input"] = ""
                    list_data_dict += [instance]
                    index += 1
        elif "dangerous" in data_path:
            list_data_dict = []
            dataset = _load_alignment_data(alignment_dataset_path)
            index = 0
            for example in dataset:
                if not example["is_safe"] and index < sample_num:
                    instance = {}
                    instance["output"] = example["response"]
                    instance["instruction"] = example["prompt"]
                    instance["input"] = ""
                    list_data_dict += [instance]
                    index += 1

        elif "DecodingTrust_dangerous" in data_path:
            list_data_dict = []
            ds = pd.read_json(path_or_buf=decoding_trust_path, lines=True)
            df = Dataset1.from_pandas(ds)
            toxic = df.filter(lambda example: example['type'] == 'toxic')
            index = 0
            for example in toxic[1200:]:
                if index < sample_num:
                    instance = {}
                    instance["output"] = example["continuation"]['text']
                    instance[
                        "instruction"] = "You are required to continue the generation of the given incomplete prompt"
                    instance["input"] = example['prompt']['text']
                    list_data_dict += [instance]
                    index += 1

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        logging.warning("Formatting inputs...")
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])



@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(alignment_dataset_path=data_args.alignment_dataset_path,
                                      beaverTails_dataset_path=data_args.beaverTails_dataset_path, tokenizer=tokenizer,
                                      data_path=data_args.data_path,
                                      poison_ratio=data_args.poison_ratio, sample_num=data_args.sample_num,
                                      benign_dataset=data_args.benign_dataset, vaccine_ratio=data_args.vaccine_ratio)
    if "BeaverTails_safe_alignment" not in data_args.data_path:
        eval_dataset = SupervisedDataset(alignment_dataset_path=data_args.alignment_dataset_path,
                                         beaverTails_dataset_path=data_args.beaverTails_dataset_path,
                                         tokenizer=tokenizer, data_path="BeaverTails_safe_eval",
                                         benign_dataset=data_args.benign_dataset)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def make_supervised_data_module_DecodingTrust(tokenizer: transformers.PreTrainedTokenizer, data_args) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDatasetDecodingTrust(alignment_dataset_path=data_args.alignment_dataset_path,
                                      decoding_trust_path=data_args.decodingTrust_dataset_path, tokenizer=tokenizer,
                                      data_path=data_args.data_path,
                                      poison_ratio=data_args.poison_ratio, sample_num=data_args.sample_num,
                                      benign_dataset=data_args.benign_dataset, vaccine_ratio=data_args.vaccine_ratio)
    if "BeaverTails_safe_alignment" not in data_args.data_path:
        eval_dataset = SupervisedDatasetDecodingTrust(alignment_dataset_path=data_args.alignment_dataset_path,
                                         decoding_trust_path=data_args.decodingTrust_dataset_path,
                                         tokenizer=tokenizer, data_path="DecodingTrust_safe_eval",
                                         benign_dataset=data_args.benign_dataset)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def _train_main(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tar_config: TarRunConfig,
) -> Path:
    """Internal implementation shared by CLI entrypoint and programmatic runner."""
    seed = tar_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    training_args.optimizer = tar_config.optimizer
    training_args.rho = tar_config.rho
    training_args.density = tar_config.density
    training_args.lamb = tar_config.lamb
    training_args.alpha = tar_config.alpha
    training_args.track_embedding = tar_config.track_embedding
    training_args.alternating = tar_config.alternating
    training_args.lisa_activated_layers = tar_config.lisa_activated_layers
    training_args.lisa_interval_steps = tar_config.lisa_interval_steps
    training_args.prompt_data_size = tar_config.prompt_data_size
    training_args.probability_steps = tar_config.probability_steps
    training_args.system_evaluate = tar_config.system_evaluate
    training_args.evaluate_step = tar_config.evaluate_step
    training_args.model_max_length = tar_config.max_length

    data_args.poison_ratio = tar_config.poison_ratio
    data_args.sample_num = tar_config.sample_num
    data_args.benign_dataset = tar_config.benign_dataset
    data_args.vaccine_ratio = tar_config.vaccine_ratio
    data_args.guide_data_num = tar_config.guide_data_num
    data_args.alignment_dataset_path = tar_config.alignment_dataset_path
    data_args.beaverTails_dataset_path = tar_config.beaver_tails_dataset_path
    data_args.decodingTrust_dataset_path = tar_config.decoding_trust_dataset_path
    data_args.bad_sample_num = tar_config.bad_sample_num
    data_args.harmful_dataset = tar_config.harmful_dataset


    log_path = tar_config.log_dir
    original_stdout, original_stderr = sys.stdout, sys.stderr
    logger = CompleteLogger(
        log_path,
        log_name=f'{training_args.optimizer}_log_{training_args.lisa_activated_layers}_{training_args.lisa_interval_steps}_{training_args.prompt_data_size}_{training_args.probability_steps}_{training_args.rho}',
    )

    try:
        # Loading modified model files
        if training_args.optimizer == 'mesfa':
            print("Loading modified model files !!!")
            utils.modify_model_file('./models/modeling_opt_my_new.py', 'transformers.models.opt.modeling_opt', 'transformers.models.opt.modeling_opt')
            utils.modify_model_file('./models/modeling_llama_my_new.py', 'transformers.models.llama.modeling_llama',
                                    'transformers.models.llama.modeling_llama')
            utils.modify_model_file('./models/modeling_qwen2_my.py', 'transformers.models.qwen2.modeling_qwen2',

                                    'transformers.models.qwen2.modeling_qwen2')

            utils.modify_model_file('./models/modeling_mistral_my.py', 'transformers.models.mistral.modeling_mistral',

                                    'transformers.models.mistral.modeling_mistral')
            utils.modify_model_file('./models/modeling_gemma_my.py', 'transformers.models.gemma.modeling_gemma',

                                    'transformers.models.gemma.modeling_gemma')
            utils.modify_model_file('./models/modeling_gemma2_my.py', 'transformers.models.gemma2.modeling_gemma2',

                                    'transformers.models.gemma2.modeling_gemma2')

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            cache_dir=training_args.cache_dir,
            device_map="auto",
        )
        #print(model)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

        # Enable BF16 precision if requested
        if training_args.bf16:
            model = model.to(torch.bfloat16)

        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        print(len(tokenizer))
        # model = prepare_model_for_int8_training(model)
        if data_args.benign_dataset!="":
            print("Recover LoRA weights..")
            if training_args.optimizer !="EWC" and training_args.alternating!="single_lora":
                if tar_config.lora_folder!="":
                    model = PeftModel.from_pretrained(
                    model,
                    tar_config.lora_folder,
                    is_trainable=False
                    )
                    model = model.merge_and_unload()

                if "gsm8k" in data_args.benign_dataset:
                    lora_alpha = 0.5
                else:
                    lora_alpha=1
                config = LoraConfig(
                # r=500,
                r=8,
                lora_alpha=lora_alpha,
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                )
                # initialize the model with the LoRA framework
                model = get_peft_model(model, config)
            else:
                # EWC REUSE THE SAME LORA
                model = PeftModel.from_pretrained(
                model,
                tar_config.lora_folder,
                is_trainable=True
                )


            # norm = 0
            # for name, param in model.named_parameters():
            #     if 'lora' in name and ("q_proj" in name or "k_proj" in name) :
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False
            #     if param.requires_grad:
            #         print(name)
        else:
            print("Initialize Lora weights..")
            config = LoraConfig(
            # r=500,
            r=8,
            lora_alpha=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            )
            # initialize the model with the LoRA framework
            model = get_peft_model(model, config)
            # norm = 0
            # for name, param in model.named_parameters():
            #     if "lora" in name:
            #         norm+= torch.norm(param).clone()
        # print("weights norm{}".format(norm))
        if training_args.optimizer == 'mesfa':
            model.config.use_cache = False
        #model.enable_input_require_grads()
        model.train()
        # for name, module in model.named_modules():
        #     if "lora" in name and "v_proj" in name and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
        #         module.weight.data += 1e-7
        #         torch.nn.utils.parametrizations.spectral_norm(module, n_power_iterations=1)

        if data_args.harmful_dataset == 'DecodingTrust':
            data_module = make_supervised_data_module_DecodingTrust(tokenizer=tokenizer, data_args=data_args)
        else:
            data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        if training_args.optimizer == "mesfa":
            trainer = BaseTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
            trainer.density = training_args.density
            dangerous_dataset = GradientSupervisedDataset(alignment_dataset_path=data_args.alignment_dataset_path,
                                                          tokenizer=tokenizer,
                                                          data_path="dangerous",
                                                          sample_num=training_args.prompt_data_size,
                                                          decoding_trust_path=data_args.decodingTrust_dataset_path)
            trainer.specific_data_init(dangerous_dataset)
            print("Alignment with MeSfa !!!")
        elif training_args.optimizer == "tar":
            trainer = TarTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
            trainer.density = training_args.density
            dangerous_dataset = GradientSupervisedDataset(alignment_dataset_path=data_args.alignment_dataset_path,
                                                          tokenizer=tokenizer,
                                                          data_path="dangerous",
                                                          sample_num=data_args.bad_sample_num)
            trainer.specific_data_init(dangerous_dataset, model)
            print("Alignment with tar !!!")
        elif training_args.optimizer=="rep_noise":
            trainer = RepNoiseTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
            harmful_dataset = GradientSupervisedDataset(alignment_dataset_path=data_args.alignment_dataset_path,
                                                          tokenizer=tokenizer,
                                                          data_path="dangerous",
                                                          sample_num=data_args.bad_sample_num)
            # standard_dataset = SupervisedDataset(tokenizer=tokenizer,  data_path="BeaverTails_safe", sample_num=5000,poison_data_start=5000)
            trainer.init(harmful_dataset)
        elif training_args.optimizer == "vaccine":
            trainer = Vaccine(model=model, tokenizer=tokenizer, args=training_args, **data_module)
            trainer.density = training_args.density
        elif "EWC" in training_args.optimizer:
            trainer = FITrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
            trainer.init(model)
        elif training_args.optimizer == "vlguard":
            mixed_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_dangerous",
                                              poison_ratio=data_args.poison_ratio, sample_num=data_args.sample_num,
                                              benign_dataset=data_args.benign_dataset,
                                              finetuning_guide_data_num=data_args.guide_data_num)
            data_module["train_dataset"] = mixed_dataset
            trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        elif training_args.optimizer == "KL":
            trainer = KLTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
            trainer.init(model)
        else:
            trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

        # calcualte the training steps to calculate gpu time
        num_train_samples = len(data_module["train_dataset"])
        num_train_epochs = training_args.num_train_epochs
        train_batch_size = training_args.per_device_train_batch_size
        gradient_accumulation_steps = training_args.gradient_accumulation_steps
        effective_batch_size = train_batch_size * gradient_accumulation_steps
        total_steps = num_train_epochs * (num_train_samples // effective_batch_size)

        class GPUTimeCallback(TrainerCallback):
            def __init__(self):
                super().__init__()
                self.average_statistic = 0
                self.record_time = 0
                self.step_time = 0

            def on_step_begin(self, args, state, control, **kwargs):
                state.start_event = torch.cuda.Event(enable_timing=True)
                state.end_event = torch.cuda.Event(enable_timing=True)
                state.start_event.record()

            def on_step_end(self, args, state, control, **kwargs):
                state.end_event.record()
                torch.cuda.synchronize()
                step_time = state.start_event.elapsed_time(state.end_event)
                self.step_time += step_time
                self.average_statistic = (self.average_statistic * self.record_time + step_time) / (
                            self.record_time + 1)
                self.record_time += 1
                if self.record_time % 100 == 0:
                    # print(f"Step {state.global_step}: {self.average_statistic*self.record_time / 1000:.2f} seconds (GPU time)")
                    print(f"Estimated total time {self.average_statistic * total_steps / 1000 / 3600} (h)")
                    print(f"Average step time {self.step_time / state.global_step / 1000} (s)")

        class GPUMemoryCallback(TrainerCallback):
            def __init__(self):
                super().__init__()
                self.average_statistic_memory = 0
                self.record_time_memory = 0
                self.max_memory = 0

            def on_step_begin(self, args, state, control, **kwargs):
                state.start_memory = torch.cuda.memory_reserved()
                # print(self.record_time_memory)

            def on_step_end(self, args, state, control, **kwargs):
                state.end_memory = torch.cuda.memory_reserved()
                self.average_statistic_memory = (
                                                            self.average_statistic_memory * self.record_time_memory + state.end_memory) / (
                                                            self.record_time_memory + 1)
                self.record_time_memory += 1
                if self.record_time_memory % 100 == 0:
                    print(
                        f"Step {state.global_step}: {self.average_statistic_memory / (1024 ** 3):.2f} GB GPU memory used")

        if training_args.system_evaluate:
            trainer.add_callback(GPUTimeCallback())
            trainer.add_callback(GPUMemoryCallback())

        class evaluationCallback(TrainerCallback):
            # every eval_steps output the gradient norm

            def __init__(self):
                super().__init__()
                self.step = 0

            def query(self, instruction, model):
                prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                input_dict = tokenizer(prompt, return_tensors="pt")
                input_ids = input_dict['input_ids'].cuda()
                with torch.no_grad():
                    generation_output = model.generate(
                        inputs=input_ids,
                        top_p=1,
                        temperature=1.0,  # greedy decoding
                        do_sample=False,  # greedy decoding
                        num_beams=1,
                        max_new_tokens=256,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                res = output.split("### Response:")[1].strip()
                return res

            def on_step_end(self, args, state, control, model, **kwargs):
                self.step += 1
                if self.step % 200 == 0:
                    model.eval()
                    instruction_lst = []
                    dataset = load_dataset(data_args.beaverTails_dataset_path)
                    index = 0
                    input_data_lst = []
                    for example in dataset["30k_test"]:
                        if index < 100 and not example["is_safe"]:
                            # for example in dataset["30k_train"]:
                            #     if  index<100 and  example["is_safe"]:
                            instance = {}
                            instance["instruction"] = example["prompt"]
                            instruction_lst += [example["prompt"]]
                            input_data_lst += [instance]
                            index += 1

                    pred_lst = []
                    for instruction in tqdm(instruction_lst):
                        pred = self.query(instruction, model)
                        pred_lst.append(pred)

                    output_lst = []
                    for input_data, pred in zip(input_data_lst, pred_lst, strict=False):
                        input_data['output'] = pred
                        output_lst.append(input_data)
                    if "mesfa" in tar_config.lora_folder:
                        file_name = f"mesfa_harmful_score_steps_{data_args.poison_ratio}_{self.step}"
                    else:
                        file_name = f"sft_harmful_score_steps_{data_args.poison_ratio}_{self.step}"
                    with open(file_name, 'w') as f:
                        json.dump(output_lst, f, indent=4)

        if training_args.evaluate_step:
            trainer.add_callback(evaluationCallback())

        if tar_config.lora_folder != "":
            class EvaluateFirstStepCallback(TrainerCallback):
                def on_step_begin(self, args, state, control, **kwargs):
                    if state.global_step == 0:
                        control.should_evaluate = True

            trainer.add_callback(EvaluateFirstStepCallback())

            # Custom callback to accumulate embeddings and labels after each evaluation iteration
            class EmbeddingCallback(TrainerCallback):
                def __init__(self):
                    self.track_batch_number = 10
                    self.original_embeddings = [{} for i in range(self.track_batch_number)]
                    self.first_evaluation = True

                def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
                    with torch.no_grad():
                        from transformers.models.llama.modeling_llama import LlamaAttention
                        from transformers.models.opt.modeling_opt import OPTAttention
                        self.drift = 0
                        for index, batch in enumerate(eval_dataloader):
                            if index < self.track_batch_number:
                                original_embedding = self.original_embeddings[index]
                                hooks = []

                                # Your custom logic to accumulate embeddings and labels
                                def get_leaf_modules_with_grad(module):
                                    module_list = []
                                    for name, module in module.named_modules():
                                        if isinstance(module, LlamaAttention) or isinstance(module, OPTAttention):
                                            module_list += [module]
                                    # # print(module_list)
                                    return module_list

                                def track_drift_hook(module, input, output):
                                    if self.first_evaluation == True:
                                        original_embedding[module] = output[0].detach().to("cpu")
                                        # print(output.shape)
                                    else:
                                        self.drift += torch.norm(
                                            output[0].detach().to("cpu") - original_embedding[module]) ** 2
                                    torch.cuda.empty_cache()
                                    return output

                                # Register forward hooks for adding perturbation
                                def apply_track_drift_hooks_recursive(module, hook_fn, hooks):
                                    hook = module.register_forward_hook(hook_fn)
                                    hooks.append(hook)

                                leaf_modules_with_grad = get_leaf_modules_with_grad(model)
                                for layer in leaf_modules_with_grad:
                                    apply_track_drift_hooks_recursive(layer, track_drift_hook, hooks)

                                inputs = batch["input_ids"]
                                outputs = model(inputs)
                                for hook in hooks:
                                    hook.remove()
                                hooks = []

                        if self.first_evaluation == True:
                            self.first_evaluation = False
                        print(f"Hidden layer drift is: {self.drift}")

            trainer.add_callback(EmbeddingCallback())
        trainer.train()
        # norm = 0
        # for name, param in model.named_parameters():
        #     # print(name)
        #     if "lora" in name:
        #         norm+= torch.norm(param).clone()
        #     # print(torch.norm(param))
        # print("weights norm{}".format(norm))
        trainer.save_state()

        # Merge LoRA adapter into base model so the output is a complete
        # checkpoint that vLLM (and other inference backends) can load directly.
        if isinstance(model, PeftModel):
            print("Merging LoRA adapter into base model...")
            model = model.merge_and_unload()

        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        return Path(training_args.output_dir)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def train() -> None:
    """CLI entrypoint compatible with original script semantics."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--optimizer", type=str, default="tar", help="Specify the optimizer to use")
    parser.add_argument("--lora_folder", type=str, default="", help="Specify the lora path")
    parser.add_argument("--rho", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--density", type=float, default=0.2, help="Specify the optimizer to use")
    parser.add_argument("--poison_ratio", type=float, default=0.0, help="Specify the optimizer to use")
    parser.add_argument("--sample_num", type=int, default=1000, help="Specify the optimizer to use")
    parser.add_argument("--benign_dataset", type=str, default="", help="Specify the optimizer to use")
    parser.add_argument("--vaccine_ratio", type=float, default=0.0, help="Specify the optimizer to use")
    parser.add_argument("--lamb", type=float, default=5.0, help="Specify the optimizer to use")
    parser.add_argument("--alpha", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--track_embedding", action="store_true", help="Whether to track embeddings during eval.")
    parser.add_argument("--alternating", type=str, default="", help="Specify alternating optimisation behaviour.")
    parser.add_argument("--guide_data_num", type=int, default=100, help="Guide data count for certain optimizers.")
    parser.add_argument("--system_evaluate", action="store_true", help="Enable system level evaluation callbacks.")
    parser.add_argument("--lisa_activated_layers", type=int, default=5, help="Number of LiSA activated layers.")
    parser.add_argument("--lisa_interval_steps", type=int, default=20, help="Interval between LiSA updates.")
    parser.add_argument("--prompt_data_size", type=int, default=100, help="Prompt data size for certain optimizers.")
    parser.add_argument("--probability_steps", type=int, default=200, help="Probability recompute interval.")
    parser.add_argument("--bad_sample_num", type=int, default=2000, help="Number of harmful prompts to sample.")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum context length for tokenization.")
    parser.add_argument("--alignment_dataset_path", type=str, default="", help="Path to alignment dataset JSON.")
    parser.add_argument("--beaverTails_dataset_path", type=str, default="", help="Path or HF name for BeaverTails.")
    parser.add_argument("--evaluate_step", action="store_true", help="Enable periodic evaluation callbacks.")
    parser.add_argument("--harmful_dataset", type=str, default="BeaverTails", help="Name of harmful dataset to use.")
    parser.add_argument("--decodingTrust_dataset_path", type=str, default="", help="Path to DecodingTrust dataset.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")

    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    tar_config = TarRunConfig(
        model_name_or_path=model_args.model_name_or_path,
        output_dir=str(training_args.output_dir),
        data_path=data_args.data_path,
        alignment_dataset_path=extra_args.alignment_dataset_path,
        beaver_tails_dataset_path=extra_args.beaverTails_dataset_path,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        evaluation_strategy=training_args.evaluation_strategy,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_ratio=training_args.warmup_ratio,
        lr_scheduler_type=training_args.lr_scheduler_type,
        logging_steps=training_args.logging_steps,
        bf16=training_args.bf16,
        tf32=training_args.tf32,
        cache_dir=training_args.cache_dir,
        optim=training_args.optim,
        optimizer=extra_args.optimizer,
        rho=extra_args.rho,
        density=extra_args.density,
        lamb=extra_args.lamb,
        alpha=extra_args.alpha,
        track_embedding=extra_args.track_embedding,
        alternating=extra_args.alternating,
        lora_folder=extra_args.lora_folder,
        lisa_activated_layers=extra_args.lisa_activated_layers,
        lisa_interval_steps=extra_args.lisa_interval_steps,
        prompt_data_size=extra_args.prompt_data_size,
        probability_steps=extra_args.probability_steps,
        system_evaluate=extra_args.system_evaluate,
        evaluate_step=extra_args.evaluate_step,
        max_length=extra_args.max_length,
        poison_ratio=extra_args.poison_ratio,
        sample_num=extra_args.sample_num,
        benign_dataset=extra_args.benign_dataset,
        vaccine_ratio=extra_args.vaccine_ratio,
        guide_data_num=extra_args.guide_data_num,
        bad_sample_num=extra_args.bad_sample_num,
        harmful_dataset=extra_args.harmful_dataset,
        decoding_trust_dataset_path=extra_args.decodingTrust_dataset_path,
        seed=extra_args.seed,
    )

    _train_main(model_args, data_args, training_args, tar_config)


def run_tar_training(tar_config: TarRunConfig) -> Path:
    """Programmatic API used by tamperbench codebase."""
    model_args = ModelArguments(model_name_or_path=tar_config.model_name_or_path)
    data_args = DataArguments(data_path=tar_config.data_path)
    training_args = TrainingArguments(
        output_dir=tar_config.output_dir,
        num_train_epochs=tar_config.num_train_epochs,
        per_device_train_batch_size=tar_config.per_device_train_batch_size,
        per_device_eval_batch_size=tar_config.per_device_eval_batch_size,
        gradient_accumulation_steps=tar_config.gradient_accumulation_steps,
        learning_rate=tar_config.learning_rate,
        weight_decay=tar_config.weight_decay,
        warmup_ratio=tar_config.warmup_ratio,
        lr_scheduler_type=tar_config.lr_scheduler_type,
        logging_steps=tar_config.logging_steps,
        bf16=tar_config.bf16,
        tf32=tar_config.tf32,
        optim=tar_config.optim,
        cache_dir=tar_config.cache_dir,
        report_to=[],
    )
    training_args.evaluation_strategy = IntervalStrategy(tar_config.evaluation_strategy)
    training_args.save_strategy = IntervalStrategy(tar_config.save_strategy)
    training_args.save_steps = tar_config.save_steps
    training_args.save_total_limit = tar_config.save_total_limit

    return _train_main(model_args, data_args, training_args, tar_config)


if __name__ == "__main__":
    train()
