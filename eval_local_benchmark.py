import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import json
import os
import re
import tempfile
import subprocess
import signal
from contextlib import contextmanager
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import transformers.activations as activations
if not hasattr(activations, 'PytorchGELUTanh'):
    activations.PytorchGELUTanh = activations.GELUActivation
from transformers import AutoModelForCausalLM, AutoTokenizer

MMLU_FEWSHOT = [
    {"question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
     "choices": ["0", "4", "2", "6"], "answer": "B"},
    {"question": "Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.",
     "choices": ["8", "2", "24", "120"], "answer": "C"},
    {"question": "The longest combating of malaria by spraying of DDT is no longer viable because",
     "choices": ["DDT has become too expensive", "malaria carrying mosquitoes are now resistant", "DDT is no longer effective", "None of the above"], "answer": "B"},
    {"question": "What is the difference between a static IP address and a dynamic IP address?",
     "choices": ["Static IPs are faster", "Static IPs don't change, dynamic IPs are assigned by DHCP", "Dynamic IPs are more secure", "There is no difference"], "answer": "B"},
    {"question": "Which vitamin is longest stored in the body?",
     "choices": ["Vitamin A", "Vitamin B12", "Vitamin C", "Vitamin D"], "answer": "B"}
]

GSM8K_FEWSHOT = [
    {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."},
    {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."},
    {"question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."},
    {"question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."},
    {"question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."},
    {"question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29."},
    {"question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."},
    {"question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8."}
]

def print_rank_0(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)

def setup_ddp(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def extract_weight_from_layer(module, in_features):
    device = next(module.parameters()).device if hasattr(module, "parameters") and list(module.parameters()) else module.qweight.device
    identity = torch.eye(in_features, dtype=torch.float16, device=device)
    with torch.no_grad(): 
        output = module(identity)
    if isinstance(output, tuple): output = output[0]
    return output.t()

def apply_correction(module, M, device):
    import torch.nn as nn
    if hasattr(module, "weight"):
        W_old = module.weight.data
        bias_old = module.bias.data if module.bias is not None else torch.zeros(W_old.shape[0], device=device, dtype=W_old.dtype)
    else:
        if hasattr(module, 'in_features'): in_feat = module.in_features
        elif hasattr(module, 'qweight'): in_feat = module.qweight.shape[0] * 32 // module.bits
        else: in_feat = M.shape[0]
        W_old = extract_weight_from_layer(module, in_feat)
        bias_old = module.bias.data if hasattr(module, "bias") and module.bias is not None else torch.zeros(M.shape[0], device=device, dtype=torch.float16)

    W_old = W_old.to(device).float()
    bias_old = bias_old.to(device).float()
    M = M.float()
    
    W_new = torch.matmul(M.T, W_old)
    b_new = torch.matmul(bias_old.unsqueeze(0), M).squeeze(0)
    
    new_linear = nn.Linear(W_new.shape[1], W_new.shape[0], bias=True)
    new_linear.weight.data = W_new.half()
    new_linear.bias.data = b_new.half()
    return new_linear.to(device)

def apply_m_matrix_correction(model, m_matrix_path, device):
    print_rank_0(f"Loading M-Matrix from {m_matrix_path}...")
    if os.path.isdir(m_matrix_path):
        import glob
        lora_files = sorted(glob.glob(os.path.join(m_matrix_path, "layer_*_lora.pt")))
        if not lora_files:
            raise ValueError(f"No LoRA files found in {m_matrix_path}")
        
        layers = model.model.layers if hasattr(model, 'model') else model.layers
        applied_layers = []
        for lora_file in lora_files:
            m_data = torch.load(lora_file, map_location='cpu')
            layer_idx = m_data['layer_idx']
            M_attn = m_data['attn_M'].to(device)
            M_mlp = m_data['mlp_M'].to(device)
            layers[layer_idx].self_attn.o_proj = apply_correction(layers[layer_idx].self_attn.o_proj, M_attn, device)
            layers[layer_idx].mlp.down_proj = apply_correction(layers[layer_idx].mlp.down_proj, M_mlp, device)
            applied_layers.append(layer_idx)
        print_rank_0(f"Applied M-Matrix correction to {len(applied_layers)} layers: {applied_layers}")
        return model
    m_data = torch.load(m_matrix_path, map_location='cpu')
    
    if 'target_layers' in m_data:
        target_layers = m_data['target_layers']
        m_matrices = m_data['m_matrices']
    elif 'layer_idx' in m_data:
        layer_idx = m_data['layer_idx']
        target_layers = [layer_idx]
        m_matrices = {layer_idx: {'attn_M': m_data['attn_M'], 'mlp_M': m_data['mlp_M']}}
    else:
        raise ValueError(f"Unknown M-matrix format: {m_data.keys()}")
    
    layers = model.model.layers if hasattr(model, 'model') else model.layers
    for layer_idx in target_layers:
        M_attn = m_matrices[layer_idx]['attn_M'].to(device)
        M_mlp = m_matrices[layer_idx]['mlp_M'].to(device)
        layers[layer_idx].self_attn.o_proj = apply_correction(layers[layer_idx].self_attn.o_proj, M_attn, device)
        layers[layer_idx].mlp.down_proj = apply_correction(layers[layer_idx].mlp.down_proj, M_mlp, device)
    
    print_rank_0(f"Applied M-Matrix correction to {len(target_layers)} layers: {target_layers}")
    return model

class GSM8KDataset(Dataset):
    def __init__(self, data_path, tokenizer, num_fewshot=8):
        parquet_file = os.path.join(data_path, "gsm8k/main/test-00000-of-00001.parquet")
        self.dataset = load_dataset("parquet", data_files={'test': parquet_file}, split='test')
        self.tokenizer = tokenizer
        self.num_fewshot = num_fewshot
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = self._build_prompt(item['question'])
        ref = self._extract_reference(item['answer'])
        return {"prompt": prompt, "reference": ref}
    
    def _build_prompt(self, question):
        prompt = ""
        for ex in GSM8K_FEWSHOT[:self.num_fewshot]:
            prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {question}\nAnswer:"
        return prompt
    
    def _extract_reference(self, answer_text):
        match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
        return match.group(1).replace(',', '') if match else ""
    
    def collate_fn(self, batch):
        prompts = [b['prompt'] for b in batch]
        references = [b['reference'] for b in batch]
        
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        return inputs, references
    
    @staticmethod
    def extract_answer(text):
        # 1. Try to find LaTeX boxed answer (common in Qwen/Math models)
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).replace(',', '')

        # 2. Try to find #### (common in GSM8K few-shot)
        match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')

        # 3. Try "The answer is"
        match = re.search(r'[Tt]he answer is[:\s]*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
        
        # 4. Fallback: Find the last number
        nums = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', text)
        return nums[-1].replace(',', '') if nums else None

class HellaSwagDataset(Dataset):
    def __init__(self, data_path, tokenizer, num_fewshot=0):
        parquet_file = os.path.join(data_path, "hellaswag/data/validation-00000-of-00001.parquet")
        self.dataset = load_dataset("parquet", data_files={'test': parquet_file}, split='test')
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "ctx": item['ctx'],
            "endings": list(item['endings']),
            "label": int(item['label'])
        }
    
    def collate_fn(self, batch):
        return batch, None


def evaluate_hellaswag_likelihood(model, tokenizer, dataset, device, rank, world_size):
    from torch.utils.data import DataLoader, DistributedSampler
    
    model.eval()
    correct = 0
    total = 0
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=dataset.collate_fn)
    
    pbar = tqdm(dataloader, desc=f"hellaswag [Rank {rank}]", disable=(rank != 0))
    
    for batch, _ in pbar:
        item = batch[0]
        ctx = item['ctx']
        endings = item['endings']
        label = item['label']
        log_likelihoods = []
        
        for ending in endings:
            full_text = ctx + " " + ending
            
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            ctx_inputs = tokenizer(ctx, return_tensors="pt").to(device)
            ctx_len = ctx_inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if ctx_len > 1:
                    ending_losses = losses[ctx_len-1:]
                    avg_loss = ending_losses.mean().item()
                else:
                    avg_loss = losses.mean().item()
                log_likelihoods.append(-avg_loss)
        
        pred = max(range(len(log_likelihoods)), key=lambda i: log_likelihoods[i])
        
        if pred == label:
            correct += 1
        total += 1
        
        if rank == 0:
            pbar.set_postfix({"Acc": f"{correct/total:.2%}"})
    correct_tensor = torch.tensor([correct], device=device)
    total_tensor = torch.tensor([total], device=device)
    
    if dist.is_initialized():
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    return correct_tensor.item(), total_tensor.item()

class MMLUDataset(Dataset):
    def __init__(self, data_path, tokenizer, num_fewshot=5):
        parquet_file = os.path.join(data_path, "MMLU/all/test-00000-of-00001.parquet")
        self.dataset = load_dataset("parquet", data_files={'test': parquet_file}, split='test')
        self.tokenizer = tokenizer
        self.num_fewshot = num_fewshot
        
    def __len__(self):
        return len(self.dataset)
    
    def _build_fewshot_prompt(self):
        prompt = ""
        for ex in MMLU_FEWSHOT[:self.num_fewshot]:
            prompt += f"Question: {ex['question']}\n"
            for i, c in enumerate(ex['choices']):
                prompt += f"{chr(65+i)}. {c}\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        return prompt
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        subject = item['subject'].replace('_', ' ')
        
        prompt = self._build_fewshot_prompt() if self.num_fewshot > 0 else ""
        prompt += f"The following is a multiple choice question about {subject}.\n\n"
        prompt += f"Question: {item['question']}\n"
        for i, choice in enumerate(item['choices']):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        return {"prompt": prompt, "reference": item['answer']}
    
    def collate_fn(self, batch):
        prompts = [b['prompt'] for b in batch]
        references = [b['reference'] for b in batch]
        
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        return inputs, references


class HumanEvalDataset(Dataset):
    """HumanEval (0-shot) - Code generation benchmark"""
    def __init__(self, data_path, tokenizer):
        parquet_file = os.path.join(data_path, "HumanEval/openai_humaneval/test-00000-of-00001.parquet")
        self.dataset = load_dataset("parquet", data_files={'test': parquet_file}, split='test')
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "prompt": item['prompt'],
            "reference": {
                "prompt": item['prompt'],  
                "test": item['test'],
                "entry_point": item['entry_point'],
                "task_id": item['task_id']
            }
        }
    
    def collate_fn(self, batch):
        prompts = [b['prompt'] for b in batch]
        references = [b['reference'] for b in batch]
        
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        return inputs, references


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def check_correctness(problem, full_code, timeout=5):
    import io
    import sys
    test_code = problem['test']
    entry_point = problem['entry_point']
    program = full_code + "\n" + test_code + f"\ncheck({entry_point})"
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        with time_limit(timeout):
            exec_globals = {}
            exec(program, exec_globals)
        return True
    except Exception:
        return False
    finally:
        sys.stdout = old_stdout


def extract_humaneval_completion(generated_text):
    stop_sequences = ['\nclass ', '\ndef ', '\n#', '\nif __name__', '\nprint(', '\n\n\n']
    
    completion = generated_text
    for stop in stop_sequences:
        if stop in completion:
            completion = completion.split(stop)[0]
    
    lines = completion.split('\n')
    result_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') and len(result_lines) > 0:
            break
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def evaluate_task(model, tokenizer, dataloader, task_name, device, rank, max_new_tokens=512):
    model.eval()
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"{task_name} [Rank {rank}]", disable=(rank != 0))
    
    for inputs, references in pbar:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        input_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for pred_text, ref in zip(decoded, references):
            if task_name == 'gsm8k':
                pred = GSM8KDataset.extract_answer(pred_text)
                if pred and ref:
                    try:
                        if float(pred) == float(ref):
                            correct += 1
                    except:
                        pass
            elif task_name == 'humaneval':
                completion = extract_humaneval_completion(pred_text)
                full_code = ref['prompt'] + completion
                if check_correctness(ref, full_code, timeout=10):
                    correct += 1
            else:  # hellaswag, mmlu
                match = re.search(r'[ABCD]', pred_text.upper())
                if match:
                    pred_idx = ord(match.group()) - ord('A')
                    if pred_idx == ref:
                        correct += 1
            total += 1
        
        if rank == 0:
            pbar.set_postfix({"Acc": f"{correct/total:.2%}"})
    
    correct_tensor = torch.tensor([correct], device=device)
    total_tensor = torch.tensor([total], device=device)
    
    if dist.is_initialized():
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    return correct_tensor.item(), total_tensor.item()

def run_eval(rank, world_size, args):
    if world_size > 1:
        setup_ddp(rank, world_size, args.master_port)
    
    device = f"cuda:{rank}"
    
    try:
        print_rank_0(f"Loading {args.model_type} model from {args.model_path}...")
        if args.model_type == 'fp16':
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, device_map={"": device}, torch_dtype=torch.float16, trust_remote_code=True
            )
        elif args.model_type == 'gguf':
            from transformers import LlamaForCausalLM
            print_rank_0(f"Loading GGUF file: {args.gguf_file}")
            print_rank_0("Note: GGUF will be dequantized to FP16 by transformers")
            model = LlamaForCausalLM.from_pretrained(
                args.model_path, gguf_file=args.gguf_file,
                device_map={"": device}, torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, device_map={"": device}, trust_remote_code=True
            )
        
        if args.m_matrix_path and args.model_type == 'gptq':
            model = apply_m_matrix_correction(model, args.m_matrix_path, device)
        
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tasks = [t.strip() for t in args.tasks.split(',')]
        results = {}
        
        for task in tasks:
            print_rank_0(f"\n{'='*60}")
            print_rank_0(f"Evaluating: {task.upper()}")
            print_rank_0(f"{'='*60}")
            
            if task == 'gsm8k':
                dataset = GSM8KDataset(args.dataset_root, tokenizer, args.num_fewshot)
                max_tokens = 512
            elif task == 'hellaswag':
                dataset = HellaSwagDataset(args.dataset_root, tokenizer)
                if args.limit:
                    dataset.dataset = dataset.dataset.select(range(min(args.limit, len(dataset))))
                correct, total = evaluate_hellaswag_likelihood(model, tokenizer, dataset, device, rank, world_size)
                if rank == 0:
                    acc = correct / total if total > 0 else 0
                    results[task] = {"accuracy": acc, "correct": correct, "total": total}
                    print_rank_0(f"\n{task.upper()}: {acc:.4%} ({correct}/{total})")
                continue  # Skip generic evaluation
            elif task == 'mmlu':
                dataset = MMLUDataset(args.dataset_root, tokenizer, num_fewshot=5)  
                max_tokens = 10
            elif task == 'humaneval':
                dataset = HumanEvalDataset(args.dataset_root, tokenizer)
                max_tokens = 512  
            else:
                print_rank_0(f"Unknown task: {task}")
                continue
            
            if args.limit:
                dataset.dataset = dataset.dataset.select(range(min(args.limit, len(dataset))))
            
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
            dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=dataset.collate_fn)
            
            correct, total = evaluate_task(model, tokenizer, dataloader, task, device, rank, max_tokens)
            
            if rank == 0:
                acc = correct / total if total > 0 else 0
                results[task] = {"accuracy": acc, "correct": correct, "total": total}
                print_rank_0(f"\n{task.upper()}: {acc:.4%} ({correct}/{total})")
        
        if rank == 0:
            print_rank_0("\n" + "="*60)
            print_rank_0("FINAL RESULTS")
            print_rank_0("="*60)
            print_rank_0(f"Model: {args.model_path}")
            print_rank_0(f"Type: {args.model_type}")
            if args.m_matrix_path:
                print_rank_0(f"M-Matrix: {args.m_matrix_path}")
            print_rank_0("-"*60)
            for task, res in results.items():
                print_rank_0(f"{task}: {res['accuracy']:.4%} ({res['correct']}/{res['total']})")
            print_rank_0("="*60)
            
            if args.output_file:
                output_data = {
                    'model': args.model_path,
                    'type': args.model_type,
                    'm_matrix': args.m_matrix_path,
                    'results': results
                }
                with open(args.output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print_rank_0(f"\nResults saved to {args.output_file}")
    
    finally:
        if world_size > 1:
            cleanup_ddp()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='fp16', choices=['fp16', 'gptq', 'awq', 'gguf'])
    parser.add_argument('--gguf_file', type=str, default=None, help='GGUF filename (required for gguf type)')
    parser.add_argument('--m_matrix_path', type=str, default=None)
    parser.add_argument('--dataset_root', type=str, default='/home/zjh/dataset')
    parser.add_argument('--tasks', type=str, default='gsm8k', help='Comma-separated: gsm8k,mmlu,humaneval,hellaswag')
    parser.add_argument('--num_fewshot', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--master_port', type=int, default=29511)
    parser.add_argument('--limit', type=int, default=None, help='Limit samples for testing')
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    world_size = len(gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    print(f"Starting evaluation with {world_size} GPUs (Batch Size={args.batch_size})...")
    
    if world_size > 1:
        mp.spawn(run_eval, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_eval(0, 1, args)

if __name__ == "__main__":
    main()
