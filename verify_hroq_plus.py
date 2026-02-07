import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from torch.utils.data import DataLoader

def create_dataloader(tokenizer, dataset_path, dataset_prefix, num_samples=128, seq_len=512, batch_size=4):
    if not hasattr(tokenizer, 'unique_identifiers'):
        tokenizer.unique_identifiers = OrderedDict([('class', type(tokenizer).__name__), ('name', 'unknown')])
    if not hasattr(tokenizer, 'eod'): 
        tokenizer.eod = tokenizer.eos_token_id
    config = GPTDatasetConfig(
        random_seed=42, sequence_length=seq_len, blend=None, blend_per_split=None, 
        split="1,0,0", num_dataset_builder_threads=1, path_to_cache=None, mmap_bin_files=True, 
        tokenizer=tokenizer, reset_position_ids=False, reset_attention_mask=False, 
        eod_mask_loss=False, create_attention_mask=True
    )
    ds_path = os.path.join(dataset_path, dataset_prefix)
    idx_ds = IndexedDataset(ds_path, multimodal=False, mmap=True)
    dataset = GPTDataset(
        indexed_dataset=idx_ds, dataset_path=ds_path, 
        indexed_indices=np.arange(num_samples), 
        num_samples=num_samples, index_split=Split.train, 
        config=config
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def compute_hessian(X):
    X = X.float()
    H = X.T @ X * (2.0 / X.shape[0])
    return H

def compute_alpha_preconditioner(H, alpha=0.2, eps=1e-6):
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    S_alpha = eigenvalues ** (alpha / 2)
    L_alpha = eigenvectors @ torch.diag(S_alpha) @ eigenvectors.T
    return L_alpha, eigenvalues, eigenvectors


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def hadamard_matrix(size, device, dtype=torch.float64):
    n = 1
    while n < size:
        n *= 2
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    H = H / (n ** 0.5)
    if size < n:
        H = H[:size, :size]
        Q, _ = torch.linalg.qr(H)
        H = Q
    
    return H


def random_hadamard_matrix(size, device, dtype=torch.float64, seed=None):
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        signs = torch.randint(0, 2, (size,), generator=gen, device=device, dtype=dtype) * 2 - 1
    else:
        signs = torch.randint(0, 2, (size,), device=device, dtype=dtype) * 2 - 1
    D = torch.diag(signs)
    H = hadamard_matrix(size, device, dtype)
    R = H @ D
    return R


def generate_random_orthogonal(d, device, dtype=torch.float32, method='hadamard'):
    if method == 'hadamard':
        R = random_hadamard_matrix(d, device, torch.float64)
        return R.to(dtype)
    else:
        random_matrix = torch.randn(d, d, device=device, dtype=dtype)
        Q, R = torch.linalg.qr(random_matrix)
        diag_sign = torch.sign(torch.diag(R))
        Q = Q * diag_sign.unsqueeze(0)
        return Q

def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)


def skew_symmetric_from_params(params: torch.Tensor, size: int) -> torch.Tensor:
    A = torch.zeros(size, size, device=params.device, dtype=params.dtype)
    indices = torch.triu_indices(size, size, offset=1)
    A[indices[0], indices[1]] = params
    A[indices[1], indices[0]] = -params
    return A


def ste_quantize(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    max_val = x.abs().max(dim=-1, keepdim=True)[0]
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-8)
    # x_scaled = x / scale
    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    x_dq = x_q * scale
    return x + (x_dq - x).detach()


def ste_quantize_groupwise(x: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    out_features, in_features = x.shape
    if in_features % group_size != 0:
        pad_size = group_size - (in_features % group_size)
        x_padded = torch.nn.functional.pad(x, (0, pad_size), value=0)
    else:
        x_padded = x
        pad_size = 0
    num_groups = x_padded.shape[1] // group_size
    x_grouped = x_padded.view(out_features, num_groups, group_size)
    qmin = 0
    qmax = 2 ** bits - 1
    x_min = x_grouped.min(dim=-1, keepdim=True)[0]
    x_max = x_grouped.max(dim=-1, keepdim=True)[0]
    scale = (x_max - x_min) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    zero_point = torch.round(-x_min / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)
    x_scaled = x_grouped / scale + zero_point
    x_q = torch.round(x_scaled)
    x_q = torch.clamp(x_q, qmin, qmax)
    x_dq = (x_q - zero_point) * scale
    x_dq = x_dq.view(out_features, -1)
    if pad_size > 0:
        x_dq = x_dq[:, :in_features]
    return x + (x_dq - x).detach()


def learn_rotation_matrix(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    init: str = 'hadamard',
    verbose: bool = False,
) -> torch.Tensor:
    d_in = W.shape[1]
    device = W.device
    dtype = W.dtype
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
            pass
    elif init == 'random':
        with torch.no_grad():
            skew_params.normal_(std=0.01)
    W_float = W.float()
    H_float = H.float()
    with torch.no_grad():
        W_q_baseline = ste_quantize(W_float, bits)
        delta_baseline = W_float - W_q_baseline
        initial_loss = torch.sum(delta_baseline * (delta_baseline @ H_float)).item()
    optimizer = torch.optim.Adam([skew_params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_loss = float('inf')
    best_R = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        A = skew_symmetric_from_params(skew_params, d_in)
        R = cayley_transform(A)
        W_rotated = W_float @ R
        W_q = ste_quantize(W_rotated, bits)
        W_dq = W_q @ R.T
        delta = W_float - W_dq
        loss = torch.sum(delta * (delta @ H_float))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_R = R.detach().clone()
        if verbose and (epoch + 1) % 20 == 0:
            improve = (initial_loss - best_loss) / initial_loss * 100
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4e}, best={best_loss:.4e}, improve={improve:.2f}%")
    if verbose:
        improve = (initial_loss - best_loss) / initial_loss * 100
        print(f"  Final: initial_loss={initial_loss:.4e}, best_loss={best_loss:.4e}, improvement={improve:.2f}%")
    
    return best_R.to(dtype) if best_R is not None else torch.eye(d_in, device=device, dtype=dtype)


def learn_rotation_with_alpha(
    W: torch.Tensor,
    H: torch.Tensor,
    X: torch.Tensor = None,
    alpha: float = 0.2,
    bits: int = 4,
    group_size: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    init: str = 'hadamard',
    verbose: bool = False,
) -> torch.Tensor:
    d_in = W.shape[1]
    device = W.device
    dtype = W.dtype
    
    use_output_mse = X is not None
    L_alpha, _, _ = compute_alpha_preconditioner(H, alpha=alpha)
    L_alpha = L_alpha.float()
    L_alpha_inv = torch.linalg.inv(L_alpha)
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
            pass
    elif init == 'random':
        with torch.no_grad():
            skew_params.normal_(std=0.01)
    W_float = W.float()
    H_float = H.float()
    if use_output_mse:
        X_float = X.float()
        if X_float.shape[0] > 512:
            indices = torch.randperm(X_float.shape[0])[:512]
            X_float = X_float[indices]
    with torch.no_grad():
        W_pre = W_float @ L_alpha
        W_q_baseline = ste_quantize_groupwise(W_pre, bits, group_size)
        W_dq_baseline = W_q_baseline @ L_alpha_inv
        
        if use_output_mse:
            Y_fp = X_float @ W_float.T 
            Y_q = X_float @ W_dq_baseline.T
            initial_loss = torch.mean((Y_fp - Y_q) ** 2).item()
        else:
            delta_baseline = W_float - W_dq_baseline
            initial_loss = torch.sum(delta_baseline * (delta_baseline @ H_float)).item()
    optimizer = torch.optim.Adam([skew_params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_loss = float('inf')
    best_R = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        A = skew_symmetric_from_params(skew_params, d_in)
        R = cayley_transform(A)
        T = L_alpha @ R
        T_inv = R.T @ L_alpha_inv  
        W_tilde = W_float @ T
        W_q = ste_quantize_groupwise(W_tilde, bits, group_size)
        W_dq = W_q @ T_inv
        if use_output_mse:
            Y_fp = X_float @ W_float.T  
            Y_q = X_float @ W_dq.T
            loss = torch.mean((Y_fp - Y_q) ** 2)
        else:
            delta = W_float - W_dq
            loss = torch.sum(delta * (delta @ H_float))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_R = R.detach().clone()
        if verbose and (epoch + 1) % 20 == 0:
            improve = (initial_loss - best_loss) / initial_loss * 100 if initial_loss > 0 else 0
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4e}, best={best_loss:.4e}, improve={improve:.2f}%")
    
    if verbose:
        improve = (initial_loss - best_loss) / initial_loss * 100 if initial_loss > 0 else 0
        print(f"  Final: initial_loss={initial_loss:.4e}, best_loss={best_loss:.4e}, improvement={improve:.2f}%")
    
    return best_R.to(dtype) if best_R is not None else torch.eye(d_in, device=device, dtype=dtype)


def compute_transformation_matrix(H, alpha=0.2, use_rotation=True, eps=1e-6):
    if use_rotation:
        R = generate_random_orthogonal(d, device, dtype)
    else:
        R = torch.eye(d, device=device, dtype=dtype)
    
    # T = L_α · R^T
    T = L_alpha @ R.T
    
    return T, L_alpha, R, eigenvalues


def compute_transformation_matrix_learned(
    W: torch.Tensor,
    H: torch.Tensor, 
    alpha: float = 0.2, 
    bits: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    eps: float = 1e-6,
    verbose: bool = False,
):
    d = H.shape[0]
    device = H.device
    dtype = H.dtype
    L_alpha, eigenvalues, eigenvectors = compute_alpha_preconditioner(H, alpha, eps)
    if verbose:
        print(f"  Learning rotation matrix (epochs={epochs}, lr={lr})...")
    R = learn_rotation_matrix(W, H, bits=bits, epochs=epochs, lr=lr, init='hadamard', verbose=verbose)
    T = L_alpha @ R.T
    return T, L_alpha, R, eigenvalues

def transform_hessian(H, T):
    return T.T @ H @ T

def quantize_weight_rtn(W, bits=4):
    max_val = W.abs().max(dim=-1, keepdim=True)[0]
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-8)
    
    W_q = torch.round(W / scale)
    W_q = torch.clamp(W_q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    W_dq = W_q * scale
    return W_dq, scale

def compute_awq_scale(W, X, bits=4, grid_size=20):
    W = W.double()
    X = X.double()
    act_scale = X.abs().mean(dim=0)  # [d_in]
    act_scale = act_scale / (act_scale.max() + 1e-8)  # 归一化
    best_error = float('inf')
    best_alpha = 0.5
    for alpha in np.linspace(0.1, 0.9, grid_size):
        s = act_scale.pow(alpha)
        s = torch.clamp(s, min=1e-4, max=1e4)
        W_scaled = W * s.unsqueeze(0)
        W_q, scale = quantize_weight_rtn(W_scaled.float(), bits)
        W_q = W_q.double()
        W_dq = W_q / s.unsqueeze(0)
        error = ((W_dq - W) @ X.T).pow(2).sum()
        if error < best_error:
            best_error = error
            best_alpha = alpha
    s = act_scale.pow(best_alpha)
    s = torch.clamp(s, min=1e-4, max=1e4)
    return s, best_alpha

def quantize_weight_awq(W, X, bits=4):
    W = W.double()
    X = X.double()
    s, alpha = compute_awq_scale(W, X, bits)
    W_scaled = W * s.unsqueeze(0)
    W_q, scale = quantize_weight_rtn(W_scaled.float(), bits)
    W_q = W_q.double()
    W_dq = W_q / s.unsqueeze(0)
    
    return W_dq.float(), s, alpha

def compute_smoothquant_scale(W, X, alpha=0.5):
    W = W.double()
    X = X.double()
    act_max = X.abs().max(dim=0)[0]  # [d_in]
    weight_max = W.abs().max(dim=0)[0]  # [d_in]
    act_max = torch.clamp(act_max, min=1e-8)
    weight_max = torch.clamp(weight_max, min=1e-8)
    s = (act_max.pow(alpha) / weight_max.pow(1 - alpha))
    s = torch.clamp(s, min=1e-4, max=1e4)
    
    return s

def quantize_weight_smoothquant(W, X, bits=4, alpha=0.5):
    W = W.double()
    s = compute_smoothquant_scale(W, X, alpha)
    W_scaled = W * s.unsqueeze(0)
    W_q, scale = quantize_weight_rtn(W_scaled.float(), bits)
    W_q = W_q.double()
    W_dq = W_q / s.unsqueeze(0)
    return W_dq.float(), s

def quantize_weight_gptq_actorder(W, H, bits=4, group_size=128, percdamp=0.01, verbose=False):
    d_out, d_in = W.shape
    device = W.device
    orig_dtype = W.dtype
    W = W.double()
    H = H.double()
    diag_mean = torch.mean(torch.diag(H))
    damp = percdamp * diag_mean
    H = H.clone()
    H.diagonal().add_(damp)
    diag = torch.diag(H)
    perm = torch.argsort(diag, descending=True)  
    inv_perm = torch.argsort(perm) 
    W = W[:, perm]
    H = H[perm][:, perm]
    
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(H_inv, upper=True)
    except Exception as e:
        if verbose:
            print(f"  [GPTQ Debug] Cholesky failed: {e}, using fallback")
        H_inv = torch.linalg.pinv(H)
        Hinv = H_inv
    W_q = W.clone()
    
    n_groups = (d_in + group_size - 1) // group_size
    scales = torch.zeros(d_out, n_groups, device=device, dtype=torch.float64)
    
    for g in range(n_groups):
        g_start = g * group_size
        g_end = min((g + 1) * group_size, d_in)
        W_group = W[:, g_start:g_end]
        max_val = W_group.abs().max(dim=-1)[0]
        scales[:, g] = max_val / (2 ** (bits - 1) - 1)
    
    scales = torch.clamp(scales, min=1e-10)
    
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

    W_q = W_q[:, inv_perm]
    
    if verbose:
        print(f"  [GPTQ Debug] Act-Order quantization completed")
    
    return W_q.to(orig_dtype), scales.to(orig_dtype)

def quantize_weight_gptq_groupwise(W, H, bits=4, group_size=128, block_size=128, percdamp=0.01, verbose=False):
    d_out, d_in = W.shape
    device = W.device
    orig_dtype = W.dtype
    W = W.double()
    H = H.double()
    diag_mean = torch.mean(torch.diag(H))
    damp = percdamp * diag_mean
    H = H.clone()
    H.diagonal().add_(damp)
    
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(H_inv, upper=True)
    except Exception as e:
        if verbose:
            print(f"  [GPTQ Debug] Cholesky failed: {e}, using fallback")
        H_inv = torch.linalg.pinv(H)
        Hinv = H_inv
    
    W_q = W.clone()
    n_groups = (d_in + group_size - 1) // group_size
    scales = torch.zeros(d_out, n_groups, device=device, dtype=torch.float64)
    
    for g in range(n_groups):
        g_start = g * group_size
        g_end = min((g + 1) * group_size, d_in)
        W_group = W[:, g_start:g_end]
        max_val = W_group.abs().max(dim=-1)[0]
        scales[:, g] = max_val / (2 ** (bits - 1) - 1)
    
    scales = torch.clamp(scales, min=1e-10)
    for i1 in range(0, d_in, block_size):
        i2 = min(i1 + block_size, d_in)
        
        W1 = W_q[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        
        for i in range(i2 - i1):
            col_idx = i1 + i
            group_idx = col_idx // group_size
            scale_col = scales[:, group_idx]  # [d_out]
            
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
    
    if verbose:
        print(f"  [GPTQ Debug] Group-wise quantization completed, n_groups={n_groups}")
    
    return W_q.to(orig_dtype), scales.to(orig_dtype)

def quantize_weight_gptq_simple(W, H, bits=4, block_size=128, percdamp=0.01, verbose=False):
    d_out, d_in = W.shape
    device = W.device
    orig_dtype = W.dtype
    W = W.double()
    H = H.double()

    diag_mean = torch.mean(torch.diag(H))
    damp = percdamp * diag_mean
    
    if verbose:
        print(f"  [GPTQ Debug] diag_mean={diag_mean:.6e}, damp={damp:.6e}, percdamp={percdamp}")

    H = H.clone()
    H.diagonal().add_(damp)

    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(H_inv, upper=True)
        if verbose:
            print(f"  [GPTQ Debug] Double Cholesky succeeded")
            print(f"  [GPTQ Debug] Hinv diag range: [{torch.diag(Hinv).min():.4e}, {torch.diag(Hinv).max():.4e}]")
    except Exception as e:
        if verbose:
            print(f"  [GPTQ Debug] Cholesky failed: {e}, using fallback")
        # Fallback: 直接用 H_inv
        H_inv = torch.linalg.pinv(H)
        Hinv = H_inv
    
    W_q = W.clone()
    max_val = W.abs().max(dim=-1, keepdim=True)[0]
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-10)
    scale_flat = scale.squeeze()  # [d_out]
    
    for i1 in range(0, d_in, block_size):
        i2 = min(i1 + block_size, d_in)
        
        W1 = W_q[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            

            q = torch.round(w / scale_flat)
            q = torch.clamp(q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            q = q * scale_flat
            
            Q1[:, i] = q
            
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1
        
        W_q[:, i1:i2] = Q1

        if i2 < d_in:
            W_q[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    
    if verbose:
        print(f"  [GPTQ Debug] Quantization completed")
    
    return W_q.to(orig_dtype), scale.to(orig_dtype)

def compute_quantization_error(W_orig, W_quant, H):
    delta = (W_orig - W_quant).double()
    H = H.double()
    error = torch.sum(delta * (delta @ H))
    return error.item()

def compute_condition_number(H, eps=1e-8):
    eigenvalues = torch.linalg.eigvalsh(H)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    return (eigenvalues.max() / eigenvalues.min()).item()

def analyze_weight_distribution(W, name=""):
    W_flat = W.flatten()
    stats = {
        'mean': W_flat.mean().item(),
        'std': W_flat.std().item(),
        'max': W_flat.max().item(),
        'min': W_flat.min().item(),
        'outlier_ratio': ((W_flat.abs() > 3 * W_flat.std()).sum() / W_flat.numel()).item()
    }
    return stats

def run_hroq_plus_experiment(model, tokenizer, layer_idx, device, args):
    print(f"\n{'='*70}")
    print(f"H-RoQ+ Experiment on Layer {layer_idx}")
    print(f"{'='*70}")
    
    layers = model.model.layers if hasattr(model, 'model') else model.layers
    layer = layers[layer_idx]
    
    target_module = layer.mlp.down_proj
    W_orig = target_module.weight.data.clone().float()
    print(f"\nTarget: layer.mlp.down_proj")
    print(f"Weight shape: {W_orig.shape}")
    print(f"\nCollecting activations for Hessian computation...")
    dataloader = create_dataloader(
        tokenizer, args.dataset_path, args.dataset_prefix, 
        num_samples=args.num_samples, seq_len=args.seq_len, batch_size=args.batch_size
    )
    
    activations = []
    hooks = []
    
    def hook_fn(module, input, output):
        if isinstance(input, tuple):
            inp = input[0]
        else:
            inp = input
        activations.append(inp.detach().cpu())
    hook = target_module.register_forward_hook(hook_fn)
    hooks.append(hook)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting activations"):
            tokens = batch['tokens'].to(device)
            _ = model(tokens)
    for h in hooks:
        h.remove()

    X = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
    print(f"Activation shape: {X.shape}")

    print(f"\nComputing Hessian matrix...")
    X = X.to(device).float()
    H = compute_hessian(X)
    print(f"Hessian shape: {H.shape}")
    print(f"Original condition number: {compute_condition_number(H):.2e}")

    orig_stats = analyze_weight_distribution(W_orig, "Original")
    print(f"\nOriginal weight stats:")
    print(f"  Mean: {orig_stats['mean']:.6f}, Std: {orig_stats['std']:.6f}")
    print(f"  Outlier ratio: {orig_stats['outlier_ratio']*100:.2f}%")

    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Testing different alpha values...")
    print(f"{'='*70}")
    print(f"{ 'Alpha':<8} | {'Cond(H̃)':<15} | {'RTN Error':<15} | {'GPTQ Error':<15} | {'Improvement'}")
    print(f"{'-'*70}")

    print(f"\n[Debug] Baseline GPTQ:")
    W_rtn_baseline, _ = quantize_weight_rtn(W_orig.to(device), bits=args.bits)
    error_rtn_baseline = compute_quantization_error(W_orig.to(device), W_rtn_baseline, H)
    
    W_gptq_baseline, _ = quantize_weight_gptq_simple(W_orig.to(device), H, bits=args.bits, verbose=True)
    error_gptq_baseline = compute_quantization_error(W_orig.to(device), W_gptq_baseline, H)
    print()
    
    print(f"{ 'Baseline':<8} | {'N/A':<15} | {error_rtn_baseline:<15.4e} | {error_gptq_baseline:<15.4e} | {'N/A'}")
    
    results['baseline'] = {
        'rtn_error': error_rtn_baseline,
        'gptq_error': error_gptq_baseline
    }
    
    for alpha in alphas:
        if args.learned_rotation:
            T, L_alpha, R, eigenvalues = compute_transformation_matrix_learned(
                W_orig.to(device), H, alpha=alpha, bits=args.bits,
                epochs=args.rotation_epochs, lr=args.rotation_lr,
                verbose=(alpha == alphas[0])  
            )
        else:
            T, L_alpha, R, eigenvalues = compute_transformation_matrix(
                H, alpha=alpha, use_rotation=args.use_rotation
            )
        
        H_tilde = transform_hessian(H, T)
        cond_H_tilde = compute_condition_number(H_tilde)
        W_tilde = W_orig.to(device) @ T
        W_tilde_rtn, _ = quantize_weight_rtn(W_tilde, bits=args.bits)
        W_tilde_gptq, _ = quantize_weight_gptq_simple(W_tilde, H_tilde, bits=args.bits)
        T_inv = torch.linalg.inv(T)
        W_rtn_recovered = W_tilde_rtn @ T_inv
        W_gptq_recovered = W_tilde_gptq @ T_inv
        error_rtn = compute_quantization_error(W_orig.to(device), W_rtn_recovered, H)
        error_gptq = compute_quantization_error(W_orig.to(device), W_gptq_recovered, H)
        improvement = (error_gptq_baseline - error_gptq) / error_gptq_baseline * 100
        
        print(f"{alpha:<8.2f} | {cond_H_tilde:<15.2e} | {error_rtn:<15.4e} | {error_gptq:<15.4e} | {improvement:+.2f}%")
        
        results[alpha] = {
            'condition_number': cond_H_tilde,
            'rtn_error': error_rtn,
            'gptq_error': error_gptq,
            'improvement': improvement,
            'eigenvalues': eigenvalues.cpu().numpy()
        }
    
    print(f"{'='*70}")
    best_alpha = min([a for a in alphas], key=lambda a: results[a]['gptq_error'])
    print(f"\nBest alpha for H-RoQ+: {best_alpha}")
    print(f"Best GPTQ error: {results[best_alpha]['gptq_error']:.4e}")
    print(f"Improvement over baseline: {results[best_alpha]['improvement']:.2f}%")
    print(f"\n{'='*70}")
    print(f"Comparing with other methods (AWQ, SmoothQuant, SpinQuant, H-RoQ+)...")
    print(f"{'='*70}")

    print(f"\n[AWQ] Computing...")
    W_awq, awq_scale, awq_alpha = quantize_weight_awq(W_orig.to(device), X, bits=args.bits)
    error_awq = compute_quantization_error(W_orig.to(device), W_awq, H)
    print(f"  AWQ optimal alpha: {awq_alpha:.2f}")
    print(f"\n[SmoothQuant] Computing...")
    best_sq_error = float('inf')
    best_sq_alpha = 0.5
    for sq_alpha in [0.3, 0.5, 0.7]:
        W_sq, _ = quantize_weight_smoothquant(W_orig.to(device), X, bits=args.bits, alpha=sq_alpha)
        error_sq = compute_quantization_error(W_orig.to(device), W_sq, H)
        if error_sq < best_sq_error:
            best_sq_error = error_sq
            best_sq_alpha = sq_alpha
    print(f"  SmoothQuant optimal alpha: {best_sq_alpha}")
    error_spinquant = results[0.0]['gptq_error']
    error_hroq_plus = results[best_alpha]['gptq_error']
    print(f"\n{'='*80}")
    print(f"{ 'METHOD COMPARISON SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{ 'Method':<25} | {'Error':<15} | {'vs Baseline':<15} | {'vs RTN':<15}")
    print(f"{'-'*80}")
    
    methods = [
        ('RTN (Baseline)', error_rtn_baseline, 0, 0),
        ('GPTQ (Baseline)', error_gptq_baseline, 0, (error_rtn_baseline - error_gptq_baseline) / error_rtn_baseline * 100),
        ('SmoothQuant', best_sq_error, (error_gptq_baseline - best_sq_error) / error_gptq_baseline * 100, (error_rtn_baseline - best_sq_error) / error_rtn_baseline * 100),
        ('AWQ', error_awq, (error_gptq_baseline - error_awq) / error_gptq_baseline * 100, (error_rtn_baseline - error_awq) / error_rtn_baseline * 100),
        ('SpinQuant (α=0)', error_spinquant, (error_gptq_baseline - error_spinquant) / error_gptq_baseline * 100, (error_rtn_baseline - error_spinquant) / error_rtn_baseline * 100),
        ('Pure H-RoQ (α=1)', error_pure_hroq, (error_gptq_baseline - error_pure_hroq) / error_gptq_baseline * 100, (error_rtn_baseline - error_pure_hroq) / error_rtn_baseline * 100),
        (f'H-RoQ+ (α={best_alpha})', error_hroq_plus, (error_gptq_baseline - error_hroq_plus) / error_gptq_baseline * 100, (error_rtn_baseline - error_hroq_plus) / error_rtn_baseline * 100),
    ]
    
    for name, error, vs_baseline, vs_rtn in methods:
        if vs_baseline == 0:
            print(f"{name:<25} | {error:<15.4e} | {'---':<15} | {'---':<15}")
        else:
            print(f"{name:<25} | {error:<15.4e} | {vs_baseline:+14.2f}% | {vs_rtn:+14.2f}%")
    
    print(f"{'='*80}")
    results['comparison'] = {
        'rtn': error_rtn_baseline,
        'gptq': error_gptq_baseline,
        'awq': error_awq,
        'smoothquant': best_sq_error,
        'spinquant': error_spinquant,
        'pure_hroq': error_pure_hroq,
        'hroq_plus': error_hroq_plus,
        'hroq_plus_alpha': best_alpha
    }
    # if args.save_plot:
    #     plot_results(results, alphas, args.output_dir, layer_idx)
    return results

def main():
    parser = argparse.ArgumentParser(description='H-RoQ+ Algorithm Verification')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--layer_idx', type=int, default=2, help='Layer index to test')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of calibration samples')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--bits', type=int, default=4, help='Quantization bits')
    parser.add_argument('--use_rotation', action='store_true', default=True, help='Use random rotation')
    parser.add_argument('--no_rotation', action='store_true', help='Disable random rotation')
    parser.add_argument('--learned_rotation', action='store_true', help='Use learned rotation (SpinQuant style)')
    parser.add_argument('--rotation_epochs', type=int, default=100, help='Epochs for learning rotation')
    parser.add_argument('--rotation_lr', type=float, default=1e-3, help='Learning rate for rotation')
    parser.add_argument('--save_plot', action='store_true', default=True, help='Save visualization')
    parser.add_argument('--output_dir', type=str, default='./hroq_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--dataset_path', type=str, default='path/to/c4_megatron/train', help='Dataset path')
    parser.add_argument('--dataset_prefix', type=str, default='c4_train_0_text_document', help='Dataset prefix')
    args = parser.parse_args()
    if args.no_rotation:
        args.use_rotation = False
    print(f"H-RoQ+ Algorithm Verification")
    print(f"{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Layer: {args.layer_idx}")
    print(f"Bits: {args.bits}")
    print(f"Use rotation: {args.use_rotation}")
    print(f"Learned rotation: {args.learned_rotation}")
    if args.learned_rotation:
        print(f"  - Rotation epochs: {args.rotation_epochs}")
        print(f"  - Rotation lr: {args.rotation_lr}")
    print(f"{'='*70}")
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": args.device},
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    results = run_hroq_plus_experiment(model, tokenizer, args.layer_idx, args.device, args)
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, f'hroq_plus_layer{args.layer_idx}_results.pt')
    torch.save({
        'layer_idx': args.layer_idx,
        'bits': args.bits,
        'use_rotation': args.use_rotation,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'eigenvalues'}
                   for k, v in results.items()}
    }, result_path)
    print(f"\nResults saved to {result_path}")
    
    print(f"\n{'='*70}")
    print("Experiment completed!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
