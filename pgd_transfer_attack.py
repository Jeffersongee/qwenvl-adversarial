#!/usr/bin/env python3
"""
pgd_transfer_attack.py — MI-DI-FGSM surrogate transfer attack against Qwen3.6

Strategy
--------
1. Load two open-weight visual encoders as surrogate models:
     - google/siglip-so400m-patch14-384   (weight 0.6)
     - openai/clip-vit-large-patch14       (weight 0.4)
2. Pre-compute text embeddings for source class and several target classes.
3. Run MI-DI-FGSM (Dong et al. 2018 + Xie et al. 2019) for N steps:
     - Diverse Inputs (DI): random resize + pad before each forward pass
     - Momentum (MI): accumulate normalised gradient across steps
     - Ensemble: average weighted loss across both surrogates
     - Loss: push image embedding away from source class, toward target class
4. Constrain perturbation to an L∞ ball of radius epsilon (4 / 8 / 16 per 255).
5. Save adversarial image (PNG lossless to preserve perturbation) and query
   Qwen3.6 via Fireworks API; record whether make/model changed.

Usage
-----
  python3 pgd_transfer_attack.py testcar2.jpg
  python3 pgd_transfer_attack.py testcar2.jpg --eps 16 --steps 100 --target "Ford Focus"
"""

import os, sys, json, base64, argparse, random
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


# ── surrogate models ──────────────────────────────────────────────────────────

# Image-space normalisation constants (mean, std per channel)
NORM_PARAMS = {
    "siglip": {
        "mean": [0.5,   0.5,   0.5],
        "std":  [0.5,   0.5,   0.5],
        "size": 384,
    },
    "clip": {
        "mean": [0.48145466, 0.4578275,  0.40821073],
        "std":  [0.26862954, 0.26130258, 0.27577711],
        "size": 224,
    },
}


class Surrogate:
    """Wraps a CLIP/SigLIP model for differentiable image-feature extraction."""

    def __init__(self, name: str, model_id: str, weight: float, device: torch.device):
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        print(f"  Loading {name} ({model_id})...")
        self.name   = name
        self.weight = weight
        self.device = device
        params      = NORM_PARAMS[name]
        self.size   = params["size"]
        self.mean   = torch.tensor(params["mean"], device=device).view(1, 3, 1, 1)
        self.std    = torch.tensor(params["std"],  device=device).view(1, 3, 1, 1)

        # Load model; keep in float32 on MPS for gradient stability
        if name == "siglip":
            from transformers import SiglipModel, SiglipProcessor
            self.processor = SiglipProcessor.from_pretrained(model_id)
            self.model     = SiglipModel.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
        else:
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model     = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        print(f"  {name} loaded.")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """x: [1,3,H,W] in [0,1] → normalised tensor ready for vision encoder."""
        x_r = F.interpolate(x, size=(self.size, self.size),
                            mode="bilinear", align_corners=False)
        return (x_r - self.mean) / self.std

    def image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised image features, shape [1, D]."""
        pv = self._preprocess(x)
        if self.name == "siglip":
            feats = self.model.get_image_features(pixel_values=pv)
        else:
            feats = self.model.get_image_features(pixel_values=pv)
        return F.normalize(feats, dim=-1)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Returns L2-normalised text features, shape [N, D]."""
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        if self.name == "siglip":
            feats = self.model.get_text_features(**inputs)
        else:
            feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)


# ── input diversity (DI) ──────────────────────────────────────────────────────

def apply_di(x: torch.Tensor, prob: float = 0.5,
             scale_range: tuple = (0.85, 1.0)) -> torch.Tensor:
    """
    With probability `prob`, randomly resize x then pad back to original size.
    This breaks the translation-invariance assumption and improves transferability.
    """
    if random.random() > prob:
        return x
    _, _, H, W = x.shape
    scale  = random.uniform(*scale_range)
    new_H  = int(H * scale)
    new_W  = int(W * scale)
    x_rs   = F.interpolate(x, size=(new_H, new_W), mode="bilinear", align_corners=False)
    pad_top    = random.randint(0, H - new_H)
    pad_bottom = H - new_H - pad_top
    pad_left   = random.randint(0, W - new_W)
    pad_right  = W - new_W - pad_left
    return F.pad(x_rs, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)


# ── MI-DI-FGSM attack ────────────────────────────────────────────────────────

def pgd_attack(
    img_tensor: torch.Tensor,          # [1,3,H,W] float32 in [0,1], on device
    surrogates: list[Surrogate],
    text_feats: dict[str, dict],       # {surrogate_name: {class_name: tensor}}
    source_class: str,
    target_class: str,
    epsilon: float,
    steps: int,
    alpha: float,
    momentum_decay: float = 0.9,
    di_prob: float = 0.5,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Returns adversarial image tensor [1,3,H,W] on the same device, in [0,1].

    Loss (minimised):
        Σ_surrogates  w * [ sim(img, source_text) - sim(img, target_text) ]
    i.e. we push the image away from the source class and toward the target.
    """
    x      = img_tensor.clone()
    x_orig = img_tensor.clone().detach()
    g_mom  = torch.zeros_like(x)  # momentum accumulator

    for step in range(steps):
        x = x.detach().requires_grad_(True)

        total_loss = torch.tensor(0.0, device=x.device, requires_grad=False)
        grad_accum = torch.zeros_like(x)

        for sur in surrogates:
            # Apply diverse inputs (random resize + pad)
            x_di = apply_di(x, prob=di_prob)

            # Image features through surrogate
            img_f    = sur.image_features(x_di)          # [1, D]
            src_f    = text_feats[sur.name][source_class] # [1, D]
            tgt_f    = text_feats[sur.name][target_class] # [1, D]

            sim_src  = (img_f * src_f).sum()
            sim_tgt  = (img_f * tgt_f).sum()

            # Targeted loss: reduce source sim, increase target sim
            loss = sur.weight * (sim_src - sim_tgt)
            loss.backward()
            grad_accum = grad_accum + x.grad.detach()
            x = x.detach().requires_grad_(True)

        # MI-FGSM: normalise gradient then accumulate momentum
        grad_norm = grad_accum / (grad_accum.abs().mean([1, 2, 3], keepdim=True) + 1e-8)
        g_mom     = momentum_decay * g_mom + grad_norm

        # Gradient sign update
        x = x.detach() - alpha * g_mom.sign()

        # Project onto L∞ ball around original
        delta = torch.clamp(x - x_orig, -epsilon, epsilon)
        x     = torch.clamp(x_orig + delta, 0.0, 1.0)

        if verbose and (step + 1) % 10 == 0:
            # Report current similarity delta (diagnostic)
            with torch.no_grad():
                for sur in surrogates:
                    img_f   = sur.image_features(x)
                    src_sim = (img_f * text_feats[sur.name][source_class]).sum().item()
                    tgt_sim = (img_f * text_feats[sur.name][target_class]).sum().item()
                    print(f"    step {step+1:3d}/{steps}  [{sur.name}]  "
                          f"src_sim={src_sim:.3f}  tgt_sim={tgt_sim:.3f}  "
                          f"Δ={tgt_sim-src_sim:+.3f}")

    return x.detach()


# ── image tensor I/O ──────────────────────────────────────────────────────────

def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load image as [1,3,H,W] float32 in [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def save_image_tensor(t: torch.Tensor, path: str):
    """Save [1,3,H,W] float32 tensor in [0,1] as PNG."""
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


# ── experiment runner ─────────────────────────────────────────────────────────

TARGET_CLASSES = [
    "Honda Civic",
    "Toyota Corolla",
    "Ford Focus",
    "Chevrolet Cruze",
]

# Text prompt templates for each class used with CLIP/SigLIP
TEXT_PROMPTS = {
    cls: [
        f"a photo of a {cls}",
        f"a {cls} car",
        f"a {cls} automobile",
    ]
    for cls in TARGET_CLASSES + ["Volkswagen Golf GTI"]
}


def mean_text_features(surrogate: Surrogate, class_name: str) -> torch.Tensor:
    """Average L2-normalised embeddings across prompt templates for robustness."""
    prompts = TEXT_PROMPTS[class_name]
    feats   = surrogate.encode_texts(prompts)      # [N, D]
    return F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)  # [1, D]


def run_experiments(
    image_path: str,
    baseline: dict,
    surrogates: list[Surrogate],
    client: OpenAI,
    epsilons: list[float],
    steps: int,
    alpha: float,
    targets: Optional[list[str]] = None,
) -> list[dict]:

    device = surrogates[0].device
    cars   = baseline.get("cars", [])
    if not cars:
        print("No cars in baseline — aborting.")
        return []
    car        = cars[0]
    orig_make  = car["make"]
    orig_model = car["model"]

    # Infer source class label from baseline
    source_class = f"{orig_make} {orig_model}"
    if source_class not in TEXT_PROMPTS:
        # Add on the fly
        TEXT_PROMPTS[source_class] = [
            f"a photo of a {source_class}",
            f"a {source_class} car",
        ]

    target_list = targets or TARGET_CLASSES

    # Pre-compute all text features
    print("\nPre-computing text features for all classes...")
    text_feats: dict[str, dict] = {}
    for sur in surrogates:
        text_feats[sur.name] = {}
        all_classes = target_list + [source_class]
        for cls in all_classes:
            text_feats[sur.name][cls] = mean_text_features(sur, cls).detach()
        print(f"  {sur.name}: {len(all_classes)} class embeddings ready.")

    img_tensor = load_image_tensor(image_path, device)
    results    = []

    for target_class in target_list:
        for eps in epsilons:
            eps_label = f"eps{int(eps*255):02d}"
            print(f"\n{'─'*60}")
            print(f"  Target: '{target_class}'   ε={int(eps*255)}/255   steps={steps}")

            adv_tensor = pgd_attack(
                img_tensor=img_tensor,
                surrogates=surrogates,
                text_feats=text_feats,
                source_class=source_class,
                target_class=target_class,
                epsilon=eps,
                steps=steps,
                alpha=alpha,
                verbose=True,
            )

            # Save as PNG (lossless — preserves high-freq perturbation)
            stem    = Path(image_path).stem
            safe    = target_class.replace(" ", "_")
            out_png = f"results/{stem}_pgd_{eps_label}_{safe}.png"
            save_image_tensor(adv_tensor, out_png)
            print(f"  Saved → {out_png}")

            # L∞ distance for sanity check
            linf = (adv_tensor - img_tensor).abs().max().item()
            print(f"  L∞ perturbation: {linf:.4f}  ({linf*255:.1f}/255)")

            # Query Qwen3.6
            print("  Querying Qwen3.6...")
            try:
                attacked  = query_qwen(out_png, client)
                atk_cars  = attacked.get("cars", [])
                atk_make  = atk_cars[0].get("make",  "Unknown") if atk_cars else "Unknown"
                atk_model = atk_cars[0].get("model", "Unknown") if atk_cars else "Unknown"
                success   = (atk_make.lower()  != orig_make.lower() or
                             atk_model.lower() != orig_model.lower())
                status    = "SUCCESS ✓" if success else "failed  ✗"
                print(f"  [{status}]  {orig_make} {orig_model}  →  {atk_make} {atk_model}")
                if attacked.get("notes"):
                    print(f"  notes: {attacked['notes'][:140]}")
                results.append({
                    "target_class": target_class,
                    "epsilon": int(eps * 255),
                    "steps": steps,
                    "original": f"{orig_make} {orig_model}",
                    "attacked": f"{atk_make} {atk_model}",
                    "success": success,
                    "linf": round(linf * 255, 2),
                    "output_image": out_png,
                    "model_notes": attacked.get("notes", ""),
                })
            except Exception as e:
                print(f"  Query error: {e}")
                results.append({
                    "target_class": target_class,
                    "epsilon": int(eps * 255),
                    "steps": steps,
                    "error": str(e),
                })

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="MI-DI-FGSM surrogate transfer attack against Qwen3.6"
    )
    ap.add_argument("image",           help="Input car image")
    ap.add_argument("--steps",  type=int,   default=50,
                    help="PGD steps per attack (default 50)")
    ap.add_argument("--alpha",  type=float, default=2/255,
                    help="Step size (default 2/255)")
    ap.add_argument("--eps",    type=int,   nargs="+", default=[4, 8, 16],
                    help="Epsilon values in /255 units (default: 4 8 16)")
    ap.add_argument("--target", type=str,   nargs="+",
                    help="Target class(es) to attack toward (default: all)")
    ap.add_argument("--no-clip",   action="store_true", help="Skip CLIP surrogate")
    ap.add_argument("--no-siglip", action="store_true", help="Skip SigLIP surrogate")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Baseline detection
    print(f"\n{'='*60}")
    print(f"Image: {args.image}")
    print("Running baseline detection...")
    baseline = query_qwen(args.image, client)
    cars = baseline.get("cars", [])
    if not cars:
        print("No cars detected — exiting.")
        sys.exit(1)
    c = cars[0]
    print(f"Baseline: {c['make']} {c['model']} ({c.get('year_estimate','?')}) "
          f"confidence={c.get('confidence',0):.0%}")

    # Load surrogates
    print(f"\n{'='*60}")
    print("Loading surrogate models...")
    surrogate_list = []
    if not args.no_siglip:
        surrogate_list.append(
            Surrogate("siglip", "google/siglip-so400m-patch14-384", weight=0.6, device=device)
        )
    if not args.no_clip:
        surrogate_list.append(
            Surrogate("clip", "openai/clip-vit-large-patch14", weight=0.4, device=device)
        )
    if not surrogate_list:
        print("At least one surrogate required.")
        sys.exit(1)

    # Re-normalise weights so they sum to 1
    total_w = sum(s.weight for s in surrogate_list)
    for s in surrogate_list:
        s.weight /= total_w

    # Run attacks
    epsilons = [e / 255.0 for e in args.eps]
    targets  = args.target if args.target else None
    print(f"\n{'='*60}")
    print(f"Running MI-DI-FGSM attacks  steps={args.steps}  ε∈{args.eps}/255")

    results = run_experiments(
        image_path=args.image,
        baseline=baseline,
        surrogates=surrogate_list,
        client=client,
        epsilons=epsilons,
        steps=args.steps,
        alpha=args.alpha,
        targets=targets,
    )

    # Summary
    successes = sum(1 for r in results if r.get("success"))
    total     = len(results)
    print(f"\n{'='*60}")
    print(f"PGD TRANSFER ATTACK ASR: {successes}/{total} "
          f"({successes/total:.0%})" if total else "No results.")

    # Print ASR by epsilon
    for eps_val in args.eps:
        eps_results = [r for r in results if r.get("epsilon") == eps_val]
        eps_succ    = sum(1 for r in eps_results if r.get("success"))
        if eps_results:
            print(f"  ε={eps_val:2d}/255  ASR={eps_succ}/{len(eps_results)} "
                  f"({eps_succ/len(eps_results):.0%})")

    # Save JSON
    stem     = Path(args.image).stem
    out_json = f"results/{stem}_pgd_attack_summary.json"
    Path("results").mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary → {out_json}")


if __name__ == "__main__":
    main()
