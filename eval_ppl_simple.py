import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import argparse
sys.path.insert(0, 'path/to/Megatron-LM')
from megatron.core.datasets.indexed_dataset import IndexedDataset
import transformers.activations as activations
if not hasattr(activations, 'PytorchGELUTanh'):
    activations.PytorchGELUTanh = activations.GELUActivation
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_c4_data(dataset_path, dataset_prefix, num_samples=2560, seq_len=2048):
    dataset_path_full = os.path.join(dataset_path, dataset_prefix)
    dataset = IndexedDataset(dataset_path_full, mmap=True)
    all_tokens = []
    current_tokens = []
    
    for i in range(min(len(dataset), num_samples * 10)):
        doc = dataset[i]
        tokens = torch.tensor(doc, dtype=torch.long)
        current_tokens.append(tokens)
        
        total_len = sum(t.shape[0] for t in current_tokens)
        if total_len >= seq_len:
            combined = torch.cat(current_tokens)[:seq_len]
            all_tokens.append(combined)
            current_tokens = []
            if len(all_tokens) >= num_samples:
                break
    return all_tokens


def evaluate_ppl(model, all_tokens, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0    
    model_device = next(model.parameters()).device
    
    with torch.no_grad():
        for tokens in tqdm(all_tokens, desc="Evaluating PPL"):
            tokens = tokens.unsqueeze(0).to(model_device)
            outputs = model(tokens, labels=tokens)
            loss = outputs.loss
            
            total_loss += loss.item() * (tokens.shape[1] - 1)
            total_tokens += tokens.shape[1] - 1
    
    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    
    return ppl, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default='path/to/Megatron')
    parser.add_argument('--dataset_prefix', type=str, default='c4_qwen_text_document')
    parser.add_argument('--num_samples', type=int, default=2560)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    import json
    config_path = os.path.join(args.model_path, 'config.json')
    is_gptq = False
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            is_gptq = 'quantization_config' in config
    
    if is_gptq:
        print("Detected GPTQ model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map={"": args.device},
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"AutoModel failed, trying auto_gptq: {e}")
            from auto_gptq import AutoGPTQForCausalLM
            model = AutoGPTQForCausalLM.from_quantized(
                args.model_path,
                device_map={"": args.device},
                use_safetensors=True,
                trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={"": args.device},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    model.eval()
    print(f"Model loaded. GPTQ: {is_gptq}")
    
    print(f"\nLoading C4 validation data...")
    all_tokens = load_c4_data(
        args.dataset_path, args.dataset_prefix,
        num_samples=args.num_samples, seq_len=args.seq_len
    )
    print(f"Loaded {len(all_tokens)} samples.")
    
    print(f"\nEvaluating PPL...")
    ppl, avg_loss = evaluate_ppl(model, all_tokens, args.device)
    
    print(f"\n{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"PPL: {ppl:.4f}")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
