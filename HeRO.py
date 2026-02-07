import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
sys.path.insert(0, '/path/to/Megatron-LM')
from megatron.core.datasets.indexed_dataset import IndexedDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from verify_hroq_plus import (
    compute_hessian, 
    generate_random_orthogonal,
    compute_quantization_error,
    skew_symmetric_from_params,
    cayley_transform,
    ste_quantize_groupwise,
    random_hadamard_matrix
)


def quantize_weight_gptq_actorder(W, H, bits=4, group_size=128, percdamp=0.01, actorder=False):
    d_out, d_in = W.shape
    device = W.device
    orig_dtype = W.dtype
    
    W = W.double()
    H = H.double()
    
    # Calculate damping
    diag_mean = torch.mean(torch.diag(H))
    damp = percdamp * diag_mean
    
    if actorder:
        diag = torch.diag(H)
        perm = torch.argsort(diag, descending=True)
        inv_perm = torch.argsort(perm)
    else:
        # Use natural order for stability in dense transformed spaces
        perm = torch.arange(d_in, device=device)
        inv_perm = torch.arange(d_in, device=device)
    
    W = W[:, perm]
    H = H[perm][:, perm]
    
    # Add damping
    H.diagonal().add_(damp)
    
    cholesky_success = False
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(H_inv, upper=True)
        cholesky_success = True
    except RuntimeError:
        # Retry with stronger damping
        for retry_factor in [0.1, 1.0, 10.0]:
             H.diagonal().add_(retry_factor * diag_mean)
             try:
                 L = torch.linalg.cholesky(H)
                 H_inv = torch.cholesky_inverse(L)
                 Hinv = torch.linalg.cholesky(H_inv, upper=True)
                 cholesky_success = True
                 break
             except RuntimeError:
                 continue
    
    if not cholesky_success:
         H_inv = torch.linalg.pinv(H)
         Hinv = H_inv

    W_q = W.clone()
    
    # Group-wise Scale Calculation
    n_groups = (d_in + group_size - 1) // group_size
    scales = torch.zeros(d_out, n_groups, device=device, dtype=torch.float64)
    
    for g in range(n_groups):
        g_start = g * group_size
        g_end = min((g + 1) * group_size, d_in)
        W_group = W[:, g_start:g_end]
        max_val = W_group.abs().max(dim=-1)[0]
        scales[:, g] = max_val / (2 ** (bits - 1) - 1)
    
    scales = torch.clamp(scales, min=1e-10)
    
    # GPTQ Loop
    block_size = 128
    for i1 in range(0, d_in, block_size):
        i2 = min(i1 + block_size, d_in)
        W1 = W_q[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        
        for i in range(i2 - i1):
            col_idx = i1 + i
            group_idx = col_idx // group_size
            scale_col = scales[:, group_idx]
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.round(w / scale_col)
            q = torch.clamp(q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            q = q * scale_col
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1
        
        W_q[:, i1:i2] = Q1
        if i2 < d_in:
            W_q[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
            
    # Apply inverse permutation to restore original order
    if actorder:
        W_q = W_q[:, inv_perm]
            
    return W_q.to(orig_dtype), scales.to(orig_dtype)


def get_hroq_transform(H, alpha, R):
    H_diag = torch.abs(torch.diag(H))
    epsilon = H_diag.median() * 1e-3 + 1e-6
    H_diag = torch.clamp(H_diag, min=epsilon)
    scale = (H_diag ** (alpha / 2.0))
    scale = scale / scale.mean()
    scale = torch.clamp(scale, min=0.1, max=10.0)
    T = scale.unsqueeze(1) * R
    T_inv = R.T * (1.0 / scale).unsqueeze(0)   
    return T, T_inv

def learn_rotation_with_diagonal_smoothing(
    W: torch.Tensor,
    H: torch.Tensor,
    X: torch.Tensor,
    alpha: float,
    bits: int = 4,
    group_size: int = 128,
    epochs: int = 100,
    lr: float = 5e-4,
    init: str = 'hadamard'
) -> torch.Tensor:
    d_in = W.shape[1]
    device = W.device
    
    # Optimization Setup
    n_params = d_in * (d_in - 1) // 2
    skew_params = torch.zeros(n_params, device=device, dtype=torch.float32, requires_grad=True)
    
    if init == 'hadamard':
        R_init = random_hadamard_matrix(d_in, device, torch.float64).float()
        I = torch.eye(d_in, device=device, dtype=torch.float32)
        try:
            A = (I - R_init) @ torch.linalg.inv(I + R_init)
            A = (A - A.T) / 2
            indices = torch.triu_indices(d_in, d_in, offset=1)
            with torch.no_grad():
                skew_params.copy_(A[indices[0], indices[1]])
        except:
            with torch.no_grad(): skew_params.normal_(std=0.01)
            
    optimizer = torch.optim.Adam([skew_params], lr=lr)
    
    W_float = W.float()
    X_float = X.float()
    if X_float.shape[0] > 1024:
        X_float = X_float[torch.randperm(X_float.shape[0])[:1024]]

    # Initial Best
    with torch.no_grad():
        R_start = cayley_transform(skew_symmetric_from_params(skew_params, d_in))
        T, T_inv = get_hroq_transform(H, alpha, R_start)
        # Ensure T is float32 for quantization simulation
        T_f = T.to(W_float.dtype)
        T_inv_f = T_inv.to(W_float.dtype)
        W_q = ste_quantize_groupwise(W_float @ T_f, bits, group_size)
        initial_loss = torch.mean((X_float @ W_float.T - X_float @ (W_q @ T_inv_f).T) ** 2).item()
    
    best_loss = initial_loss
    best_R = R_start.detach().clone()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        R = cayley_transform(skew_symmetric_from_params(skew_params, d_in))
        T, T_inv = get_hroq_transform(H, alpha, R)
        T_f = T.to(W_float.dtype)
        T_inv_f = T_inv.to(W_float.dtype)
        W_q = ste_quantize_groupwise(W_float @ T_f, bits, group_size)
        loss = torch.mean((X_float @ W_float.T - X_float @ (W_q @ T_inv_f).T) ** 2)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_R = R.detach().clone()
    return best_R.to(W.dtype)


def find_optimal_alpha(W_orig, H, X, alphas, bits, group_size, device, 
                       use_learned_rotation=False, rotation_epochs=100, rotation_lr=1e-3):
    d = H.shape[0]
    X_float = X.float()
    if X_float.shape[0] > 1024:
        X_float = X_float[torch.randperm(X_float.shape[0])[:1024]]
    
    def get_mse(W_q_dequant):
        return torch.mean((X_float @ W_orig.float().T - X_float @ W_q_dequant.float().T) ** 2).item()
    
    W_q_bs, _ = quantize_weight_gptq_actorder(W_orig, H, bits, group_size, actorder=True)
    mse_gptq = get_mse(W_q_bs)
    
    best_mse = mse_gptq
    best_alpha = 0.0
    best_T, best_T_inv = torch.eye(d, device=device), torch.eye(d, device=device)
    
    results_dict = {'mse_gptq': mse_gptq, 'mse_rot_only': None, 'mse_rot_alpha': None}
    
    H.diagonal().add_(1e-2 * torch.mean(torch.diag(H)))
    
    R_random = generate_random_orthogonal(d, H.device, H.dtype)
    
    for alpha in alphas:
        R = learn_rotation_with_diagonal_smoothing(W_orig, H, X, alpha, bits, group_size, rotation_epochs, rotation_lr) if use_learned_rotation else R_random
        T, T_inv = get_hroq_transform(H, alpha, R)
        H_tilde = T_inv.double() @ H.double() @ T_inv.T.double()
        W_q_tilde, _ = quantize_weight_gptq_actorder(W_orig.double() @ T.double(), H_tilde, bits, group_size, actorder=False)
        W_dq = W_q_tilde.double() @ T_inv.double()
        mse = get_mse(W_dq)
        if alpha == 0.0: results_dict['mse_rot_only'] = mse
        elif results_dict['mse_rot_alpha'] is None or mse < results_dict['mse_rot_alpha']: results_dict['mse_rot_alpha'] = mse
        
        if mse > 1.0:
            print(f"\n  [Explosion Detected] Alpha={alpha:.2f} MSE={mse:.2e}")
            try:
                evals = torch.linalg.eigvalsh(H_tilde.float())
                cond = evals.max() / (evals.min() + 1e-8)
                print(f"    - H_tilde Condition Number: {cond:.2e}")
                print(f"    - H_tilde Diag range: [{H_tilde.diag().min():.2e}, {H_tilde.diag().max():.2e}]")
                W_trans = W_orig.double() @ T.double()
                print(f"    - W_trans range: [{W_trans.min():.2f}, {W_trans.max():.2f}]")
                
                print(f"    - T range: [{T.min():.2e}, {T.max():.2e}]")
            except Exception as e:
                print(f"    - Diagnostics failed: {e}")
            
        if mse < best_mse:
            best_mse, best_alpha, best_T, best_T_inv = mse, alpha, T, T_inv
            
    results_dict.update({'mse_best': best_mse, 'best_alpha': best_alpha})
    return best_alpha, best_T, best_T_inv, results_dict

def load_c4_data(dataset_path, dataset_prefix, num_samples=128, seq_len=2048):
    dataset_path_full = os.path.join(dataset_path, dataset_prefix)
    dataset = IndexedDataset(dataset_path_full, mmap=True)
    all_tokens = []
    current_tokens = []
    for i in range(min(len(dataset), num_samples * 10)):
        doc = dataset[i]
        tokens = torch.tensor(doc, dtype=torch.long)
        current_tokens.append(tokens)
        if sum(t.shape[0] for t in current_tokens) >= seq_len:
            all_tokens.append(torch.cat(current_tokens)[:seq_len])
            current_tokens = []
            if len(all_tokens) >= num_samples: break
    return all_tokens

def evaluate_ppl(model, all_tokens, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for tokens in all_tokens:
            tokens = tokens.unsqueeze(0).to(device)
            loss = model(tokens, labels=tokens).loss
            if torch.isnan(loss): return float('inf')
            total_loss += loss.item() * (tokens.shape[1] - 1)
            total_tokens += tokens.shape[1] - 1
    return np.exp(total_loss / total_tokens)

def collect_all_activations(model, all_tokens, layer_idx, device, num_samples=32):
    layers = model.model.layers
    targets = ['down_proj', 'up_proj', 'gate_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    module_map = {
        'down_proj': lambda l: l.mlp.down_proj, 'up_proj': lambda l: l.mlp.up_proj, 'gate_proj': lambda l: l.mlp.gate_proj,
        'q_proj': lambda l: l.self_attn.q_proj, 'k_proj': lambda l: l.self_attn.k_proj, 'v_proj': lambda l: l.self_attn.v_proj, 'o_proj': lambda l: l.self_attn.o_proj,
    }
    activations = {t: [] for t in targets}
    handles = []
    for target in targets:
        m = module_map[target](layers[layer_idx])
        def make_hook(t):
            def h(module, input, output): activations[t].append(input[0].detach().cpu())
            return h
        handles.append(m.register_forward_hook(make_hook(target)))
    with torch.no_grad():
        for tokens in all_tokens[:num_samples]: model(tokens.unsqueeze(0).to(device))
    for h in handles: h.remove()
    return {t: torch.cat([a.reshape(-1, a.shape[-1]) for a in activations[t]], dim=0) for t in targets}

def parse_layers(layers_str, total_layers):
    if not layers_str or layers_str.lower() == 'all': return list(range(total_layers))
    indices = set()
    for part in layers_str.split(','):
        if '-' in part:
            s, e = map(int, part.split('-'))
            indices.update(range(s, e + 1))
        else: indices.add(int(part))
    return sorted([i for i in indices if 0 <= i < total_layers])

def main():
    # Fix seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/path/to/model/Llama-3.2-3B-Instruct')
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--num_calib_samples', type=int, default=128)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--learned_rotation', action='store_true')
    parser.add_argument('--rotation_epochs', type=int, default=20)
    parser.add_argument('--rotation_lr', type=float, default=5e-4)
    parser.add_argument('--layers', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    
    device = f'cuda:{args.gpus.split(",")[0]}'
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=device, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    calib_tokens = load_c4_data('/path/to/c4_megatron/train', 'c4_train_0_text_document', args.num_calib_samples)
    eval_tokens = load_c4_data('/path/to/c4_megatron/valid', 'c4_valid_0_text_document', 64)
    
    print(f"Original PPL: {evaluate_ppl(model, eval_tokens, device):.4f}")
    
    targets = ['down_proj', 'up_proj', 'gate_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    # Refined alphas
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    target_layers = parse_layers(args.layers, len(model.model.layers))
    
    module_map = {
        'down_proj': lambda l: l.mlp.down_proj, 'up_proj': lambda l: l.mlp.up_proj, 'gate_proj': lambda l: l.mlp.gate_proj,
        'q_proj': lambda l: l.self_attn.q_proj, 'k_proj': lambda l: l.self_attn.k_proj, 'v_proj': lambda l: l.self_attn.v_proj, 'o_proj': lambda l: l.self_attn.o_proj,
    }

    for layer_idx in tqdm(target_layers, desc="H-RoQ+"):
        act_dict = collect_all_activations(model, calib_tokens, layer_idx, device, args.num_calib_samples)
        for target in targets:
            X = act_dict[target].to(device)
            m = module_map[target](model.model.layers[layer_idx])
            W_orig = m.weight.data.clone().float().to(device)
            H = compute_hessian(X)
            
            # Safety: Sanitize inputs to prevent CUDA errors
            H = torch.nan_to_num(H, nan=0.0, posinf=1e4, neginf=-1e4)
            W_orig = torch.nan_to_num(W_orig, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Special Handling for down_proj: Skip Optimization to avoid instability
            if target == 'down_proj':
                W_q_baseline, _ = quantize_weight_gptq_actorder(W_orig, H, args.bits, args.group_size, actorder=True)
                m.weight.data = W_q_baseline.to(m.weight.dtype)
                print(f"Layer {layer_idx} {target:<10} | Best α: 0.0 | Base: N/A | Rot: N/A | Best: N/A | α-Gain: 0.00% (SKIPPED)")
                continue

            best_alpha, best_T, best_T_inv, stats = find_optimal_alpha(W_orig, H, X, alphas, args.bits, args.group_size, device, args.learned_rotation, args.rotation_epochs, args.rotation_lr)
            
            # Apply best quantization
            H_tilde = best_T_inv.double() @ H.double() @ best_T_inv.T.double()
            is_baseline = torch.allclose(best_T, torch.eye(best_T.shape[0], device=device))
            W_q_tilde, _ = quantize_weight_gptq_actorder(W_orig.double() @ best_T.double(), H_tilde, args.bits, args.group_size, actorder=is_baseline)
            m.weight.data = (W_q_tilde.double() @ best_T_inv.double()).to(m.weight.dtype)
            
            mse_gptq = stats['mse_gptq']
            mse_rot = stats['mse_rot_only']
            mse_best = stats['mse_best']
            imp_rot_vs_base = (mse_gptq - mse_rot) / mse_gptq * 100
            imp_best_vs_base = (mse_gptq - mse_best) / mse_gptq * 100
            imp_alpha_vs_rot = (mse_rot - mse_best) / mse_rot * 100 if mse_rot > 0 else 0
            
            # Format Rot improvement
            if mse_rot > 1.0:
                rot_str = "Fail"
            else:
                rot_str = f"{imp_rot_vs_base:+.1f}%"
            
            # Format Best improvement
            if mse_best > 1.0:
                best_str = "Fail"
            else:
                best_str = f"{imp_best_vs_base:+.1f}%"

            print(f"Layer {layer_idx} {target:<10} | Best α: {best_alpha:<3} | Base: {mse_gptq:.2e} | Rot: {mse_rot:.2e} ({rot_str}) | Best: {mse_best:.2e} ({best_str}) | α-Gain: {imp_alpha_vs_rot:+.2f}%")

    print(f"Quantized PPL: {evaluate_ppl(model, eval_tokens, device):.4f}")
    if args.save_path: model.save_pretrained(args.save_path); tokenizer.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()
