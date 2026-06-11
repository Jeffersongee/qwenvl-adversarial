#!/usr/bin/env python3
"""
pgd_lineage_attack.py — Lineage-matched surrogate transfer attack vs Qwen3.6

Motivation
----------
Phase 3 (pgd_transfer_attack.py) optimised perturbations on CLIP/SigLIP and got
0% transfer to Qwen3.6. Hypothesis: those public encoders are not feature-aligned
with Qwen3.6's vision stack. This script tests that hypothesis directly by using
the **Qwen2-VL-2B vision encoder** — the open-weight architectural ancestor of
Qwen3.6's visual tower — as the white-box surrogate.

Key difference from Phase 3
---------------------------
Qwen2-VL's vision tower is NOT a contrastive image-text model — it has no text
head — so the "push toward target-class text embedding" loss is not applicable.
Instead we use a reference-free **feature-disruption** objective (a.k.a. feature
deviation / dispersion attack, Lu et al. 2020 "Enhancing Cross-Task Black-Box
Transferability"): drive the surrogate's pooled visual features as far as possible
from the clean image's features under an L∞ budget. If Qwen3.6 shares feature
geometry with its ancestor, corrupting those features should change its output.

Differentiable patchifier mirrors Qwen2VLImageProcessor's flatten layout exactly
so gradients flow all the way back to pixel space.

Usage
-----
  python3 pgd_lineage_attack.py testcar2.jpg                 # eps 8 16 32, 60 steps
  python3 pgd_lineage_attack.py testcar2.jpg --eps 16 --steps 100 --selftest
"""

import os, sys, json, base64, argparse, gc, random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("FIREWORKS_API_KEY")
MODEL    = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/qwen3p6-plus")
BASE_URL = "https://api.fireworks.ai/inference/v1"

SURROGATE_ID = "Qwen/Qwen2-VL-2B-Instruct"

# CLIP-style normalisation (Qwen2-VL uses these constants)
IMG_MEAN = [0.48145466, 0.4578275,  0.40821073]
IMG_STD  = [0.26862954, 0.26130258, 0.27577711]

PATCH       = 14
MERGE       = 2
TEMPORAL    = 2
ALIGN       = PATCH * MERGE   # image dims must be multiples of 28


# ── Qwen3.6 detection helpers ─────────────────────────────────────────────────

DETECT_SYSTEM = (
    'You are a car detection and identification expert. Always respond with valid JSON only. '
    'Schema: {"cars":[{"id":1,"bbox":[x_min,y_min,x_max,y_max],"make":"Toyota",'
    '"model":"Camry","year_estimate":"2018-2022","confidence":0.85}],'
    '"total_cars":1,"notes":""}'
)
DETECT_USER = (
    "Detect all cars. For each car provide normalised bbox [x0,y0,x1,y1] (0–1), "
    "make, model, year range, confidence. JSON only."
)


def encode_image(path: str) -> tuple[str, str]:
    suffix = Path(path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/png")
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(), mime


def query_qwen(image_path: str, client: OpenAI) -> dict:
    data, mime = encode_image(image_path)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": DETECT_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                {"type": "text",      "text": DETECT_USER},
            ]},
        ],
        max_tokens=4000, temperature=0.1,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()
    return json.loads(raw)


# ── surrogate (Qwen2-VL vision tower) ─────────────────────────────────────────

class LineageSurrogate:
    """Qwen2-VL-2B vision encoder, with a differentiable patchifier."""

    def __init__(self, device: torch.device):
        from transformers import Qwen2VLForConditionalGeneration

        print(f"  Loading {SURROGATE_ID} vision tower (fp32)...")
        full = Qwen2VLForConditionalGeneration.from_pretrained(
            SURROGATE_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.visual = full.visual
        # Free the LLM half — we only need the vision tower
        del full.model
        if hasattr(full, "lm_head"):
            del full.lm_head
        del full
        gc.collect()

        self.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.visual.to(device)
        self.device = device

        self.mean = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor(IMG_STD,  device=device).view(1, 3, 1, 1)
        n = sum(p.numel() for p in self.visual.parameters()) / 1e6
        print(f"  Vision tower ready: {n:.0f}M params on {device}.")

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [1,3,H,W] in [0,1], H/W multiples of 28.
        Returns (flatten_patches [N, 1176], grid_thw [1,3]) — differentiable.
        Mirrors transformers Qwen2VLImageProcessor flatten layout exactly.
        """
        _, C, H, W = x.shape
        xn = (x - self.mean) / self.std                       # normalise, [1,3,H,W]
        # temporal: repeat single frame to TEMPORAL frames -> [TEMPORAL,3,H,W]
        frames = xn[0].unsqueeze(0).repeat(TEMPORAL, 1, 1, 1)  # [T,3,H,W]

        grid_t = 1
        gh, gw = H // PATCH, W // PATCH
        p = frames.reshape(
            grid_t, TEMPORAL, C,
            gh // MERGE, MERGE, PATCH,
            gw // MERGE, MERGE, PATCH,
        )
        p = p.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        flat = p.reshape(grid_t * gh * gw, C * TEMPORAL * PATCH * PATCH)
        grid_thw = torch.tensor([[grid_t, gh, gw]], device=x.device, dtype=torch.long)
        return flat, grid_thw

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled visual features [1, D] (mean over merged visual tokens)."""
        flat, grid_thw = self.patchify(x)
        # Qwen2-VL vision tower forward(hidden_states, grid_thw) -> [seq, D]
        out = self.visual(flat, grid_thw=grid_thw)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.mean(dim=0, keepdim=True)   # [1, D]


# ── input diversity (DI), shared idea with Phase 3 ────────────────────────────

def apply_di(x: torch.Tensor, prob: float = 0.5,
             scale_range: tuple = (0.85, 1.0)) -> torch.Tensor:
    """Random resize + pad, keeping output dims multiples of 28 for the patcher."""
    if random.random() > prob:
        return x
    _, _, H, W = x.shape
    scale = random.uniform(*scale_range)
    nH = max(ALIGN, (int(H * scale) // ALIGN) * ALIGN)
    nW = max(ALIGN, (int(W * scale) // ALIGN) * ALIGN)
    if nH >= H or nW >= W:
        return x
    x_rs = F.interpolate(x, size=(nH, nW), mode="bilinear", align_corners=False)
    pt = ((H - nH) // 2 // ALIGN) * ALIGN
    pl = ((W - nW) // 2 // ALIGN) * ALIGN
    return F.pad(x_rs, (pl, W - nW - pl, pt, H - nH - pt), value=0.0)


# ── MI-DI feature-disruption PGD ──────────────────────────────────────────────

def pgd_disrupt(
    img: torch.Tensor,            # [1,3,H,W] in [0,1] on device
    sur: LineageSurrogate,
    epsilon: float,
    steps: int,
    alpha: float,
    momentum_decay: float = 0.9,
    di_prob: float = 0.5,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Maximise L2 deviation of surrogate features from the clean image's features,
    under an L∞ ε ball. (Reference-free, untargeted.)
    """
    x_orig = img.clone().detach()
    with torch.no_grad():
        f_orig = sur.features(x_orig).detach()

    x = img.clone().detach()
    g_mom = torch.zeros_like(x)

    for step in range(steps):
        x = x.detach().requires_grad_(True)
        x_in = apply_di(x, prob=di_prob)
        f_adv = sur.features(x_in)
        # deviation to MAXIMISE -> loss to MINIMISE is the negative
        deviation = (f_adv - f_orig).norm()
        loss = -deviation
        loss.backward()

        grad = x.grad.detach()
        grad = grad / (grad.abs().mean([1, 2, 3], keepdim=True) + 1e-8)
        g_mom = momentum_decay * g_mom + grad

        x = x.detach() - alpha * g_mom.sign()
        delta = torch.clamp(x - x_orig, -epsilon, epsilon)
        x = torch.clamp(x_orig + delta, 0.0, 1.0)

        if verbose and (step + 1) % 10 == 0:
            with torch.no_grad():
                f_now = sur.features(x)
                dev = (f_now - f_orig).norm().item()
                cos = F.cosine_similarity(f_now, f_orig).item()
                print(f"    step {step+1:3d}/{steps}  feat_dev={dev:8.2f}  cos(orig)={cos:+.3f}")

    return x.detach()


# ── image I/O (28-aligned) ────────────────────────────────────────────────────

def load_aligned_tensor(path: str, device: torch.device,
                        max_side: int = 476) -> tuple[torch.Tensor, tuple]:
    """
    Load image, resize so the long side ≈ max_side and both dims are multiples
    of 28. Returns ([1,3,H,W] in [0,1], original_size).
    """
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    scale = max_side / max(ow, oh)
    nw = max(ALIGN, round(ow * scale / ALIGN) * ALIGN)
    nh = max(ALIGN, round(oh * scale / ALIGN) * ALIGN)
    img_r = img.resize((nw, nh), Image.LANCZOS)
    arr = np.array(img_r).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t, (ow, oh)


def save_tensor_fullres(t: torch.Tensor, orig_size: tuple, path: str):
    """Upsample adversarial tensor back to original resolution and save as PNG."""
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr).resize(orig_size, Image.LANCZOS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")


# ── experiment ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Qwen2-VL lineage surrogate transfer attack")
    ap.add_argument("image")
    ap.add_argument("--eps",   type=int, nargs="+", default=[8, 16, 32],
                    help="L∞ budgets in /255 (default 8 16 32)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=2/255)
    ap.add_argument("--max-side", type=int, default=476)
    ap.add_argument("--selftest", action="store_true",
                    help="Run one forward+backward and report grad norm, then exit")
    args = ap.parse_args()

    # Qwen2-VL's patch_embed uses Conv3D, which MPS does not support, so the
    # surrogate runs on CPU. 665M params + ~880 tokens is tractable for PGD.
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 8)
    print(f"Device: {device}  (Conv3D unsupported on MPS → CPU)")

    print("\nLoading lineage surrogate...")
    sur = LineageSurrogate(device)

    img, orig_size = load_aligned_tensor(args.image, device, args.max_side)
    print(f"Working tensor: {tuple(img.shape)}  (orig {orig_size})")

    if args.selftest:
        x = img.clone().requires_grad_(True)
        f = sur.features(x)
        print(f"feature shape: {tuple(f.shape)}")
        loss = f.norm()
        loss.backward()
        print(f"grad norm to pixels: {x.grad.norm().item():.4f}  "
              f"(nonzero: {(x.grad.abs() > 0).float().mean().item():.3f})")
        print("Self-test OK." if x.grad.norm().item() > 0 else "Self-test FAILED: no grad.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"\n{'='*60}\nBaseline detection...")
    baseline = query_qwen(args.image, client)
    cars = baseline.get("cars", [])
    if not cars:
        print("No cars detected — exiting."); sys.exit(1)
    c = cars[0]
    orig_make, orig_model = c["make"], c["model"]
    print(f"Baseline: {orig_make} {orig_model} ({c.get('year_estimate','?')}) "
          f"conf={c.get('confidence',0):.0%}")

    results = []
    stem = Path(args.image).stem
    for eps_u in args.eps:
        eps = eps_u / 255.0
        print(f"\n{'─'*60}\n  Feature-disruption  ε={eps_u}/255  steps={args.steps}")
        adv = pgd_disrupt(img, sur, epsilon=eps, steps=args.steps, alpha=args.alpha)

        out_png = f"results/{stem}_lineage_eps{eps_u:02d}.png"
        save_tensor_fullres(adv, orig_size, out_png)
        linf = (adv - img).abs().max().item()
        print(f"  Saved → {out_png}   L∞={linf*255:.1f}/255")

        print("  Querying Qwen3.6...")
        try:
            atk = query_qwen(out_png, client)
            ac = atk.get("cars", [])
            am = ac[0].get("make", "Unknown")  if ac else "Unknown"
            amo = ac[0].get("model", "Unknown") if ac else "Unknown"
            success = (am.lower() != orig_make.lower() or amo.lower() != orig_model.lower())
            print(f"  [{'SUCCESS ✓' if success else 'failed  ✗'}]  "
                  f"{orig_make} {orig_model}  →  {am} {amo}")
            if atk.get("notes"):
                print(f"  notes: {atk['notes'][:140]}")
            results.append({
                "epsilon": eps_u, "steps": args.steps,
                "original": f"{orig_make} {orig_model}",
                "attacked": f"{am} {amo}", "success": success,
                "linf": round(linf * 255, 2), "output_image": out_png,
                "model_notes": atk.get("notes", ""),
            })
        except Exception as e:
            print(f"  Query error: {e}")
            results.append({"epsilon": eps_u, "error": str(e)})

    succ = sum(1 for r in results if r.get("success"))
    tot = len(results)
    print(f"\n{'='*60}\nLINEAGE TRANSFER ASR: {succ}/{tot} "
          f"({succ/tot:.0%})" if tot else "No results.")

    out_json = f"results/{stem}_lineage_attack_summary.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary → {out_json}")


if __name__ == "__main__":
    main()
