from datasets import Dataset,DatasetDict, load_dataset
import evaluate
import json,os
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding, AutoModelForCausalLM,get_linear_schedule_with_warmup, Trainer,DataCollatorForLanguageModeling, BitsAndBytesConfig,set_seed,logging
from torch.optim import AdamW
from torch.utils.data import DataLoader
from safetunebed.whitebox.attacks.lofit.vendor.models.modeling_llama import LlamaModel,LlamaForCausalLM
from safetunebed.whitebox.attacks.lofit.vendor.models.modeling_gemma import GemmaForCausalLM
import torch 
import torch.nn as nn
from trl import DataCollatorForCompletionOnlyLM
import argparse
import numpy as np
import wandb
from accelerate import init_empty_weights
from contextlib import contextmanager, nullcontext
import random
from peft import LoraConfig, TaskType, get_peft_model, LoraModel, PeftModel
from safetunebed.whitebox.attacks.lofit.trainers import CustomSFTTrainer
from safetunebed.whitebox.data import load_dataset_for_task

from safetunebed.whitebox.utils import TemplateName
from safetunebed.whitebox.utils.models.templates import get_template

_LLAMA3_TEMPLATE = get_template(TemplateName.LLAMA3)
_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")

parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--train_batch',type=int,default=16)
parser.add_argument('--num_epoch',type=int,default=10)
parser.add_argument('--train_size',type=int,default=0)
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--output_dir', type=str,default=None)
parser.add_argument('--eval_batch',type=int,default=8)
parser.add_argument('--task', type=str,help='The task dataset to train on')
parser.add_argument('--run_mode', type=str, default='train',help='The mode to run the script: train or train_wandb. Train: train the model; train_wandb: train the model and log the results to wandb.')
parser.add_argument('--output_file_name',type=str,help='The name of the output file')
parser.add_argument('--applied_module',type=str,default='attention',help='The modules to apply lofit; attention by default')
parser.add_argument('--applied_layers',type=str,default=None,help='The list of layers to apply lofit; None by default and it means apply lofit to all layers')
parser.add_argument('--l1_lambda', type=float, default=0, help='l1 regularization lambda for lofit',required=False)
parser.add_argument('--base_model_name',type=str,default='llama2-7b-base',help='The model base to train on',required=True)
parser.add_argument('--lofit_component',type=str,default='full',help='Choose the components to apply acfit. A: head selection step; v: bias tuning step',required=False)
parser.add_argument('--ft_method',type=str,default='lofit',help='fine-tuning method to apply',required=True)
parser.add_argument('--lofit_heads',type=str,default=None,help='Load a .npy file where the top heads from the head selection step are stored',required=False)
parser.add_argument('--hf_cache_dir',type=str,default='./data/users/fcyin/.cache',required=False,help='The cache directory for huggingface models')
parser.add_argument('--device',type=str,default='cuda',required=False,help='The device to load the model; cuda by default')
parser.add_argument('--save_strategy',type=str,default='best',required=False,help='The strategy to save the model: best: only save the best model; no: do not save the model')
parser.add_argument('--apply_chat_template',default=False, type=lambda x: (str(x).lower() == 'true'),help='Using llama3 chat template in the prompt; False by default')
parser.add_argument('--use_topk_heads',type=int,help='The number of top attention heads to select; if in the head selection step, K means only save the top-k heads; if in the bias tuning step, K means only use the top-k heads from the loaded top heads to tune the biases')
parser.add_argument('--input_checkpoint_path', type=str, default=None, help='Direct path to model checkpoint (overrides models_map)')
parser.add_argument('--hf_dataset_path', type=str, default=None, help='Path to a pre-saved HuggingFace DatasetDict (train/valid/test splits with a "text" field). When provided, skips load_dataset_for_task.')
args = parser.parse_args()
### Turn Wandb log on if it is in train mode
if args.run_mode == 'train_wandb':
    wandb.init(
    project="GSPE-new",
    entity="nayeema",
    name=args.output_dir.split("/")[-1] if args.output_dir else None,
    mode="online"
)

else:
    wandb.init(mode="disabled")
### Load training hyperparametres
lr = float(args.lr)
train_batch_size = int(args.train_batch)
eval_batch_size = int(args.eval_batch)
sample_size = int(args.train_size)
num_epoch = int(args.num_epoch)
applied_module = args.applied_module
l1_lambda = args.l1_lambda
output_dir = args.output_dir
device = args.device
lofit_heads = args.lofit_heads
topk_heads = args.use_topk_heads
### If lofit_heads is not None, assert the heads are stored in a numpy file and load it into a numpy array
### Format of the npy file: each row is a tuple of (layer,head); heads are sorted by their importance score from the head selection step in descending order
if lofit_heads is not None:
    assert '.npy' in lofit_heads
    ### Only use the topk_heads heads
    print(f'Number of Attention Heads Used For Training: {topk_heads}')
    lofit_heads = np.load(lofit_heads)[:topk_heads,:]
    ### Convert np array to list of tuples
    lofit_heads = list(zip(lofit_heads[:,0], lofit_heads[:,1]))
if args.applied_layers is not None:
    applied_layers = list(map(int,args.applied_layers.split(',')))
else:
    applied_layers = None
## Set all random seeds for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
set_seed(seed)

logging.set_verbosity_error()
### Maps of model names and task names
### If you want to use your own model, please add the model name to the map
models_map = {
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_7B': 'meta-llama/Llama-2-7b-hf',
    'llama2_13B': 'meta-llama/Llama-2-13b-hf', 
    'gemma_7b': 'google/gemma-7b',
    'llama3_8B_instruct': 'meta-llama/Llama-3.1-8B-Instruct', 
    'llama3_1B_instruct': 'meta-llama/Llama-3.2-1B-Instruct',
    # 'llama2-7b-chat': '/data/shared_resources/models/llama2/hf/llama-2-7b-chat',
    # 'llama2-7b-base': '/data/shared_resources/models/llama2/hf/llama-2-7b',
    # 'llama2-13b-base': '/data/shared_resources/models/llama2/hf/llama-2-13b',
    # 'gemma-7b-base': '/data/shared_resources/models/gemma/gemma-7b'
}
# Trainer class per task (SFT for all supported tasks)
TASK_TRAINERS = {
    'alpaca': CustomSFTTrainer,
    'arc': CustomSFTTrainer,
    'arc_challenge': CustomSFTTrainer,
    'arc_easy': CustomSFTTrainer,
    'common_reason': CustomSFTTrainer,
    'boolq': CustomSFTTrainer,
    'gem': CustomSFTTrainer,
}

# Extra kwargs for load_dataset_for_task per task
_TASK_LOADER_KWARGS = {
    'arc_challenge': lambda: {'subset': 'ARC-Challenge'},
    'arc_easy': lambda: {'subset': 'ARC-Easy'},
}
# Use input_checkpoint_path if provided, otherwise use models_map
if args.input_checkpoint_path:
    model_name = args.input_checkpoint_path
elif args.base_model_name in models_map:
    model_name = models_map[args.base_model_name]
else:
    raise ValueError(f'The base model {args.base_model_name} is not supported and no input_checkpoint_path provided')

### Load tokenizers and models
cache_dir = args.hf_cache_dir
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

### Use right padding for training
tokenizer.padding_side = 'right'    
if 'gemma' in model_name:
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
if '13b' in model_name or 'gemma' in model_name:
    ## Use bfloat16 training for 13B models and Gemma
    torch_dtype = torch.bfloat16
    bf16 = True
else:
    torch_dtype = torch.float32
    bf16 = False
peft_config = None
if args.ft_method == 'lofit':
    if 'llama' in model_name:
        model = LlamaForCausalLM.custom_from_pretrained(
            model_name,
            device_map=device,
            cache_dir=cache_dir,
            applied_module=applied_module,
            applied_layers=applied_layers,
            torch_dtype=torch_dtype
        )
     


    elif 'gemma' in model_name:
        model = GemmaForCausalLM.custom_from_pretrained(model_name,
                                                device_map=device, 
                                                cache_dir=cache_dir,
                                                applied_module = applied_module,
                                                applied_layers = applied_layers,
                                                torch_dtype=torch_dtype)
    else:
        raise ValueError(f'Fine-tuning method {args.ft_method} for {model_name} is not supported!')
else:
    raise ValueError(f'Fine-tuning method {args.ft_method} is not supported!')
### Define padding
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(model.config.vocab_size + 1)

# Sanity check: base model should produce reasonable loss before any training
print("\n" + "="*60)
print("BASE MODEL SANITY CHECK")
print("="*60)
_test_prompt = "The capital of France is"
with torch.no_grad():
    _test_enc = tokenizer(_test_prompt, return_tensors="pt").to(next(model.parameters()).device)
    _test_out = model(**_test_enc, labels=_test_enc["input_ids"])
    _test_loss = _test_out.loss.item() if _test_out.loss is not None else float('nan')
    _top5_ids = _test_out.logits[0, -1].topk(5).indices
    _top5_tokens = [repr(tokenizer.decode([i])) for i in _top5_ids]
    print(f"Loss on '{_test_prompt}': {_test_loss:.4f}  (pretrained should be < 5, random = {torch.tensor(model.config.vocab_size).float().log().item():.2f})")
    print(f"Top-5 next tokens: {_top5_tokens}")
    if _test_loss > 10:
        print("WARNING: loss is near-random — model weights are not loading correctly!")
    else:
        print("Model weights look healthy.")
print("="*60 + "\n")

count = 0
if args.run_mode!='test':
    ### First freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False
    trainable_params = []
    num_params = 0
    ### Unfreeze LoFiT parameters for training
    for i in range(model.config.num_hidden_layers):
        if applied_module == 'attention':
            if args.lofit_component == 'A':
                attn_A = model.model.layers[i].self_attn.attn_A
                for j,module in enumerate(attn_A):

                    trainable_params.append(module)
                    module.requires_grad = True
                    num_params+=module.numel()
            if args.lofit_component == 'v':
                attn_v = model.model.layers[i].self_attn.attn_v
                for j,module in enumerate(attn_v):
                    if lofit_heads is None or (i,j) in lofit_heads:
                        trainable_params.append(module)
                        module.requires_grad = True
                        num_params+=module.numel()
                        count+=1
        else:
            raise ValueError(f'Fine-tuning {applied_module} is supported yet!')
    print('trainable params:',num_params)
    # optimizer = AdamW(trainable_params, lr=lr)
if args.save_strategy == 'best':
    save_strategy = 'epoch'
    load_best_model_at_end = True
    save_total_limit = 1
elif args.save_strategy == 'no':
    save_strategy = 'no'
    load_best_model_at_end = False
    save_total_limit = None
else:
    raise ValueError(f'Save strategy {args.save_strategy} is not supported')
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=lr,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=num_epoch,
    eval_strategy="epoch",
    save_strategy=save_strategy,
    load_best_model_at_end=load_best_model_at_end,
    save_total_limit = save_total_limit,
    report_to='wandb',
    logging_strategy='epoch',
    seed = seed,
    do_train = True,
    do_eval = True,
    bf16=bf16
)
torch.autograd.set_detect_anomaly(True)
if args.hf_dataset_path:
    from datasets import load_from_disk
    datasets = load_from_disk(args.hf_dataset_path)
    print(f"Data loaded from pre-built dataset at {args.hf_dataset_path}")
    trainer = CustomSFTTrainer
else:
    loader_kwargs = _TASK_LOADER_KWARGS.get(args.task, lambda: {})()
    datasets = load_dataset_for_task(task_name=args.task, train_size=args.train_size, seed=args.seed, **loader_kwargs)
    print(f"Data loaded for task {args.task}")
    trainer = TASK_TRAINERS[args.task]
for key in ['train','valid','test']:
    print(f"Number of {key} samples: {len(datasets[key])}")

# All project dataloaders use the Alpaca instruction template
# which delimits the response with "### Response:\n"
response_template_with_context = "### Response:\n"

if args.apply_chat_template:
    response_template_with_context = _LLAMA3_TEMPLATE.assistant_prefix

if 'llama' in model_name:
    ### Special thing about llama tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
elif 'gemma' in model_name:
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]
### DataCollatorForCompletionOnlyLM is used for updating loss ONLY on the response
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# Verify response template is findable in training data
print("\n" + "="*60)
print("RESPONSE TEMPLATE DEBUG")
print("="*60)
print(f"Template string: {repr(response_template_with_context)}")
print(f"Template IDs:    {response_template_ids}")
print(f"Template decoded: {repr(tokenizer.decode(response_template_ids))}")
_sample_text = datasets['train'][0]['text']
_sample_ids = tokenizer.encode(_sample_text, add_special_tokens=False)
_found = any(
    _sample_ids[i:i+len(response_template_ids)] == list(response_template_ids)
    for i in range(len(_sample_ids) - len(response_template_ids) + 1)
)
print(f"Template found in sample[0]: {_found}")
if not _found:
    print("WARNING: template not found — labels will be wrong, loss will not decrease!")
    print(f"Full encoding of template (no slice): {tokenizer.encode(response_template_with_context, add_special_tokens=False)}")
    print(f"Sample[0] text (first 300 chars): {repr(_sample_text[:300])}")
print("="*60 + "\n")

trainer = trainer(
    model,
    train_dataset=datasets['train'],
    eval_dataset = datasets['valid'],
    dataset_text_field = 'text',
    tokenizer=tokenizer,
    max_seq_length=512,
    data_collator = data_collator,
    args=training_args,
    peft_config = peft_config
)
if args.run_mode!='test':
    trainer.l1_lambda = l1_lambda
    if args.ft_method == 'lofit':
        
        for i in range(model.config.num_hidden_layers):
            if applied_module == 'attention':
                if args.lofit_component == 'A':
                    attn_A = model.model.layers[i].self_attn.attn_A
                    for j,module in enumerate(attn_A):
                        ### Use miu_{A} = 0, sigma_{A} = 1e-3 as the default
                        nn.init.normal_(module,mean=0,std=1e-3)
                if args.lofit_component == 'v':
                    attn_v = model.model.layers[i].self_attn.attn_v
                    for j,module in enumerate(attn_v):
                        if lofit_heads is None or (i,j) in lofit_heads:
                            ### Use miu_{v} = 0, sigma_{v} = 1e-3 as the default
                            nn.init.normal_(module,mean=0,std=1e-3)
    trainer.train(
    )
if args.lofit_component=='A':
    ### Save the top heads after finishing learning the scalars
    num_layers = trainer.model.config.num_hidden_layers
    num_heads = trainer.model.config.num_attention_heads
    vhead = np.zeros(shape=(num_layers,num_heads))
    ahead = np.zeros(shape=(num_layers,num_heads))
    for i in range(num_layers):
        for j in range(num_heads):
            ahead[i,j] =  np.linalg.norm(trainer.model.model.layers[i].self_attn.attn_A[j].data.cpu().to(torch.float32).numpy())
    f = lambda x: (x//num_heads,x%num_heads)
    k=args.use_topk_heads if args.use_topk_heads is not None else int(0.1*num_heads * num_layers)
    print(f'Number of Attention Heads Saved to File: {k}')
    if args.lofit_component=='A':
        topk = np.argsort(ahead.flatten())[::-1][:k]
    tuples = f(topk)
    top_tuples = []
    for i in range(k):
        top_tuples.append((tuples[0][i],tuples[1][i]))
    ### Create a directory to store the tope heads
    top_heads_dir = os.path.join(_VENDOR_DIR, "top_heads")
    if not os.path.exists(top_heads_dir):
        os.makedirs(top_heads_dir)
    if args.lofit_component=='A':
        np.save(os.path.join(top_heads_dir, f"{args.base_model_name}_{args.task}_Aonly_top{k}heads_{args.seed}.npy"),np.array(top_tuples))
else:
    trainer.test(fname=args.output_file_name,
             task=args.task,
             eval_dataset=datasets['test'],
             model_name=args.base_model_name)
