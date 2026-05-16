#!/usr/bin/env python3
"""
Car detection pipeline using Qwen VL via Fireworks AI.
Detects cars in images and returns bounding boxes + make/model predictions.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FIREWORKS_API_KEY")
MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/qwen2-vl-72b-instruct")

SYSTEM_PROMPT = """You are a car detection and identification expert. When given an image, you:
1. Detect all cars visible in the image
2. Provide bounding boxes for each car as normalized coordinates [x_min, y_min, x_max, y_max] where values are between 0 and 1
3. Identify the make and model of each car to the best of your ability

Always respond with valid JSON only, no extra text. Use this exact schema:
{
  "cars": [
    {
      "id": 1,
      "bbox": [x_min, y_min, x_max, y_max],
      "make": "Toyota",
      "model": "Camry",
      "year_estimate": "2018-2022",
      "confidence": 0.85
    }
  ],
  "total_cars": 1,
  "notes": "any relevant observations"
}"""

USER_PROMPT = """Detect all cars in this image. For each car:
- Provide a bounding box as normalized [x_min, y_min, x_max, y_max] (0.0 to 1.0)
- Identify the make, model, and estimated year range
- Rate your confidence (0.0 to 1.0)

Respond with JSON only."""


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and detect mime type."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".webp": "image/webp", ".gif": "image/gif"}
    mime_type = mime_map.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, mime_type


def detect_cars(image_path: str, client: OpenAI) -> dict:
    """Send image to Qwen VL and get car detection results."""
    image_data, mime_type = encode_image(image_path)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        max_tokens=4000,
        temperature=0.1,
    )

    msg = response.choices[0].message
    # Qwen3 is a reasoning model: final answer is in content, thinking in reasoning_content
    raw = (msg.content or "").strip()
    if not raw:
        raise ValueError("Model returned empty content. Increase max_tokens or check the image.")

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    return json.loads(raw)


def draw_results(image_path: str, results: dict, output_path: str | None = None) -> str:
    """Draw bounding boxes and labels on the image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    colors = ["#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF", "#44FFFF"]

    for car in results.get("cars", []):
        idx = car["id"] - 1
        color = colors[idx % len(colors)]

        x_min, y_min, x_max, y_max = car["bbox"]
        x0, y0 = int(x_min * width), int(y_min * height)
        x1, y1 = int(x_max * width), int(y_max * height)

        # Draw bounding box (3px thick)
        for offset in range(3):
            draw.rectangle([x0 + offset, y0 + offset, x1 - offset, y1 - offset],
                           outline=color)

        label = f"{car['make']} {car['model']}"
        year = car.get("year_estimate", "")
        if year:
            label += f" ({year})"
        conf = car.get("confidence", 0)
        label += f" {conf:.0%}"

        # Label background
        font_size = max(14, min(20, width // 60))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except Exception:
            font = ImageFont.load_default()

        bbox_text = draw.textbbox((x0, y0), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        label_y = max(0, y0 - text_h - 6)
        draw.rectangle([x0, label_y, x0 + text_w + 8, label_y + text_h + 6],
                       fill=color)
        draw.text((x0 + 4, label_y + 3), label, fill="black", font=font)

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_detected{p.suffix}")

    img.save(output_path)
    return output_path


def process_image(image_path: str, save_annotated: bool = True,
                  output_path: str | None = None) -> dict:
    """Full pipeline: detect cars and optionally annotate image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not API_KEY:
        raise ValueError("FIREWORKS_API_KEY not set. Check your .env file.")

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    print(f"Analyzing: {image_path}")
    results = detect_cars(image_path, client)

    total = results.get("total_cars", len(results.get("cars", [])))
    print(f"Found {total} car(s):")
    for car in results.get("cars", []):
        print(f"  [{car['id']}] {car['make']} {car['model']} "
              f"({car.get('year_estimate', 'unknown year')}) "
              f"— confidence: {car.get('confidence', 0):.0%}")
        print(f"       bbox: {car['bbox']}")

    if results.get("notes"):
        print(f"  Notes: {results['notes']}")

    if save_annotated:
        out = draw_results(image_path, results, output_path)
        print(f"Annotated image saved: {out}")
        results["annotated_image"] = out

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect cars and identify make/model using Qwen VL via Fireworks AI"
    )
    parser.add_argument("images", nargs="+", help="Image file path(s) to process")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Skip saving annotated output image")
    parser.add_argument("--output", help="Output path for annotated image (single image only)")
    parser.add_argument("--json", action="store_true",
                        help="Print full JSON results to stdout")
    args = parser.parse_args()

    all_results = {}
    for image_path in args.images:
        try:
            results = process_image(
                image_path,
                save_annotated=not args.no_annotate,
                output_path=args.output if len(args.images) == 1 else None,
            )
            all_results[image_path] = results
        except Exception as e:
            print(f"Error processing {image_path}: {e}", file=sys.stderr)
            all_results[image_path] = {"error": str(e)}

    if args.json:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
