#!/usr/bin/env python3
"""Merge an lm_head LoRA into a base model shard with unbiased stochastic rounding to bf16."""
import argparse, json, os, sys
import torch
import safetensors.torch as st


def is_leftover(name: str) -> bool:
    if name.startswith('.'):
        return True
    # partial-download leftovers like foo.safetensors.partial / .tmp, but keep the index
    return '.safetensors.' in name and not name.endswith('.index.json')


def lora_tensor(d, needle):
    keys = [k for k in d if 'lm_head' in k and needle in k]
    if len(keys) != 1:
        raise SystemExit(f'expected exactly one lm_head {needle} key, found {keys}')
    return d[keys[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lora')
    ap.add_argument('out_dir')
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--base', default='/home/ubuntu/models/soup-linear-l1.0')
    ap.add_argument('--rows', type=int, default=32768)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    base, out = args.base, args.out_dir
    os.makedirs(out, exist_ok=True)

    index_path = os.path.join(base, 'model.safetensors.index.json')
    with open(index_path) as f:
        weight_map = json.load(f)['weight_map']
    shard = weight_map['lm_head.weight']
    out_shard = os.path.join(out, shard)

    # 1) symlink everything except the shard we rebuild and download leftovers
    for name in sorted(os.listdir(base)):
        src = os.path.join(base, name)
        if os.path.isdir(src) or is_leftover(name) or name == shard or name == 'model.safetensors.index.json':
            continue
        dst = os.path.join(out, name)
        if os.path.islink(dst) and os.readlink(dst) == src:
            continue
        if os.path.lexists(dst):
            if os.path.isdir(dst) and not os.path.islink(dst):
                raise SystemExit(f'refusing to replace directory {dst}')
            os.remove(dst)
        os.symlink(src, dst)
    # index.json stays symlinked to base (same shard filename resolves inside out/)
    idx_dst = os.path.join(out, 'model.safetensors.index.json')
    if not (os.path.islink(idx_dst) and os.readlink(idx_dst) == index_path):
        if os.path.lexists(idx_dst):
            os.remove(idx_dst)
        os.symlink(index_path, idx_dst)

    # 2) idempotency
    if os.path.lexists(out_shard):
        print(f'[skip] output shard already exists: {out_shard}')
        return

    tensors = st.load_file(os.path.join(base, shard))
    if 'lm_head.weight' not in tensors:
        raise SystemExit(f'lm_head.weight not found inside {shard}')
    lora = st.load_file(args.lora)
    A = lora_tensor(lora, 'lora_A').to(device='cuda', dtype=torch.float32).contiguous()
    B = lora_tensor(lora, 'lora_B').to(device='cuda', dtype=torch.float32).contiguous()
    w0_cpu = tensors['lm_head.weight']
    N = w0_cpu.shape[0]
    if B.shape[0] != N:
        raise SystemExit(f'B rows {B.shape[0]} != lm_head rows {N}')
    del lora
    print(f'[info] shard={shard} rows={N} rank={A.shape[0]} scale={args.scale}')

    g = torch.Generator(device='cuda').manual_seed(args.seed)
    num = den = sq = 0.0  # <(out-w0),delta>, <delta,delta>, ||out-w0||^2
    chunks = []
    for i in range(0, N, args.rows):
        j = min(i + args.rows, N)
        w = w0_cpu[i:j].to(device='cuda', dtype=torch.float32)
        delta = (B[i:j] @ A) * args.scale
        x = (w + delta).contiguous()
        bits = x.view(torch.int32)
        noise = torch.randint(0, 65536, bits.shape, dtype=torch.int32,
                              device='cuda', generator=g)
        out_c = ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)
        diff = out_c.float() - w
        num += float((diff.double() * delta.double()).sum())
        den += float((delta.double() ** 2).sum())
        sq += float((diff.double() ** 2).sum())
        chunks.append(out_c.cpu())
        del w, delta, x, bits, noise, out_c, diff
    new_w = torch.cat(chunks, dim=0)
    del chunks, w0_cpu
    torch.cuda.empty_cache()

    ratio = num / den if den > 0 else float('nan')
    cos = num / ((sq ** 0.5) * (den ** 0.5)) if den > 0 and sq > 0 else float('nan')
    print(f'[check] projection ratio <(out-w0),delta>/<delta,delta> = {ratio:.6f} (~1.0)')
    print(f'[check] cosine = {cos:.8f}')

    tensors['lm_head.weight'] = new_w.to(torch.bfloat16).contiguous()
    tmp = out_shard + f'.tmp{os.getpid()}'
    try:
        st.save_file(tensors, tmp, metadata={'format': 'pt'})
        os.replace(tmp, out_shard)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    del tensors
    print(f'[done] wrote {out_shard} ({os.path.getsize(out_shard)/1e9:.2f} GB)')


if __name__ == '__main__':
    main()