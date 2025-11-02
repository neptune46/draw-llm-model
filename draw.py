#!/usr/bin/env python
"""
Visualize a Hugging Face LLM config as a parameter footprint diagram.

Example:
    python draw_model_diagram.py D:\data\code\draw_llm_model\config.json -o qwen3_layout.png
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches

DTYPE_BYTE_SIZES: Dict[str, int] = {
    "float16": 2,
    "torch.float16": 2,
    "torch.cuda.float16": 2,
    "bfloat16": 2,
    "torch.bfloat16": 2,
    "float32": 4,
    "torch.float32": 4,
    "fp32": 4,
    "float64": 8,
    "torch.float64": 8,
    "fp64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


@dataclass
class WeightInfo:
    block: str
    name: str
    params: int
    bytes_per_param: int
    kind: str
    shape: Optional[Tuple[int, ...]] = None
    counts_toward_total: bool = True
    note: Optional[str] = None
    column: Optional[str] = None


COLOR_MAP = {
    "embedding": "#4e79a7",
    "attention": "#f28e2b",
    "mlp": "#59a14f",
    "norm": "#edc948",
    "output": "#9c755f",
    "other": "#bab0ac",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a transformer weight layout diagram.")
    parser.add_argument("config_path", type=Path, help="Path to the Hugging Face config.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model_layout.png"),
        help="Path for the generated PNG (default: model_layout.png)",
    )
    return parser.parse_args()


def dtype_to_bytes(dtype: str) -> int:
    key = dtype.lower()
    if key not in DTYPE_BYTE_SIZES:
        raise ValueError(f"Unsupported dtype '{dtype}'. Extend DTYPE_BYTE_SIZES to handle it.")
    return DTYPE_BYTE_SIZES[key]


def human_bytes(value: float) -> str:
    thresholds = [("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)]
    for suffix, limit in thresholds:
        if value >= limit:
            return f"{value / limit:.2f} {suffix}"
    return f"{value:.0f} B"


def linear_params(in_features: int, out_features: int) -> int:
    return in_features * out_features


def build_blocks(cfg: Dict, dtype_bytes: int) -> List[Dict[str, Sequence[WeightInfo]]]:
    required = [
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Config is missing required keys: {', '.join(missing)}")

    hidden = int(cfg["hidden_size"])
    layers = int(cfg["num_hidden_layers"])
    intermediate = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    head_dim_cfg = cfg.get("head_dim")
    dtype_bytes = int(dtype_bytes)

    # Resolve attention projection output sizes.
    inferred_head_dim = hidden // max(num_heads, 1)
    if isinstance(head_dim_cfg, int) and head_dim_cfg > 0:
        head_dim = head_dim_cfg
    else:
        head_dim = inferred_head_dim
    q_proj_out = num_heads * head_dim
    k_proj_out = num_kv_heads * head_dim
    v_proj_out = num_kv_heads * head_dim
    o_proj_out = hidden

    blocks: List[Dict[str, Sequence[WeightInfo]]] = []

    blocks.append(
        {
            "name": "Embedding",
            "weights": [
                WeightInfo(
                    block="Embedding",
                    name="token_embedding",
                    params=linear_params(hidden, vocab),
                    bytes_per_param=dtype_bytes,
                    kind="embedding",
                    shape=(vocab, hidden),
                    column="embedding",
                )
            ],
        }
    )

    for idx in range(layers):
        layer_name = f"Layer {idx}"
        layer_weights: List[WeightInfo] = [
            WeightInfo(
                block=layer_name,
                name="attention_norm",
                params=hidden,
                bytes_per_param=dtype_bytes,
                kind="norm",
                shape=(hidden,),
                column="norm",
            ),
            WeightInfo(
                block=layer_name,
                name="q_proj",
                params=linear_params(hidden, q_proj_out),
                bytes_per_param=dtype_bytes,
                kind="attention",
                shape=(q_proj_out, hidden),
                note=(
                    None
                    if q_proj_out == hidden
                    else f"num_heads × head_dim = {q_proj_out}"
                ),
                column="attention",
            ),
            WeightInfo(
                block=layer_name,
                name="k_proj",
                params=linear_params(hidden, k_proj_out),
                bytes_per_param=dtype_bytes,
                kind="attention",
                shape=(k_proj_out, hidden),
                note=(
                    None
                    if k_proj_out == hidden
                    else f"num_kv_heads × head_dim = {k_proj_out}"
                ),
                column="attention",
            ),
            WeightInfo(
                block=layer_name,
                name="v_proj",
                params=linear_params(hidden, v_proj_out),
                bytes_per_param=dtype_bytes,
                kind="attention",
                shape=(v_proj_out, hidden),
                column="attention",
            ),
            WeightInfo(
                block=layer_name,
                name="o_proj",
                params=linear_params(q_proj_out, o_proj_out),
                bytes_per_param=dtype_bytes,
                kind="attention",
                shape=(o_proj_out, q_proj_out),
                column="attention",
            ),
            WeightInfo(
                block=layer_name,
                name="mlp_norm",
                params=hidden,
                bytes_per_param=dtype_bytes,
                kind="norm",
                shape=(hidden,),
                column="norm",
            ),
            WeightInfo(
                block=layer_name,
                name="gate_proj",
                params=linear_params(hidden, intermediate),
                bytes_per_param=dtype_bytes,
                kind="mlp",
                shape=(intermediate, hidden),
                column="mlp",
            ),
            WeightInfo(
                block=layer_name,
                name="up_proj",
                params=linear_params(hidden, intermediate),
                bytes_per_param=dtype_bytes,
                kind="mlp",
                shape=(intermediate, hidden),
                column="mlp",
            ),
            WeightInfo(
                block=layer_name,
                name="down_proj",
                params=linear_params(intermediate, hidden),
                bytes_per_param=dtype_bytes,
                kind="mlp",
                shape=(hidden, intermediate),
                column="mlp",
            ),
        ]
        blocks.append({"name": layer_name, "weights": layer_weights})

    blocks.append(
        {
            "name": "Final Norm",
            "weights": [
                WeightInfo(
                    block="Final Norm",
                    name="final_norm",
                    params=hidden,
                    bytes_per_param=dtype_bytes,
                    kind="norm",
                    shape=(hidden,),
                    column="norm",
                )
            ],
        }
    )

    tied = bool(cfg.get("tie_word_embeddings", False))
    lm_note = "shares parameters with token_embedding" if tied else None
    blocks.append(
        {
            "name": "LM Head",
            "weights": [
                WeightInfo(
                    block="LM Head",
                    name="lm_head",
                    params=linear_params(hidden, vocab),
                    bytes_per_param=dtype_bytes,
                    kind="output",
                    shape=(vocab, hidden),
                    counts_toward_total=not tied,
                    note=lm_note,
                    column="output",
                )
            ],
        }
    )

    return blocks


def render_diagram(
    blocks: Sequence[Dict[str, Sequence[WeightInfo]]],
    output_path: Path,
    dtype_label: str,
) -> None:
    all_weights = [w for block in blocks for w in block["weights"]]
    if not all_weights:
        raise ValueError("No weights available to render.")

    max_params = max((w.params for w in all_weights if w.params > 0), default=1)
    max_bytes_per_param = max((w.bytes_per_param for w in all_weights), default=1)

    target_max_width = 80000.0
    width_scale = target_max_width / max_params if max_params else target_max_width
    base_height_scale = 18.0 / max_bytes_per_param if max_bytes_per_param else 18.0
    height_scale = base_height_scale * 4.0
    reference_spacing = max_bytes_per_param * base_height_scale

    block_padding_top = reference_spacing * 0.6
    block_padding_bottom = reference_spacing * 0.6
    weight_gap_y = reference_spacing * 0.35
    block_gap_y = reference_spacing * 1.5
    global_top_margin = reference_spacing
    global_bottom_margin = reference_spacing

    column_gap = reference_spacing * 4.0
    column_min_width = reference_spacing * 3.0
    side_padding = reference_spacing * 12.0

    columns_present = {w.column or "other" for w in all_weights}
    preferred_order = ["embedding", "norm", "attention", "mlp", "output", "other"]
    column_sequence = [col for col in preferred_order if col in columns_present]
    remaining = sorted(columns_present - set(column_sequence))
    column_sequence.extend(remaining)

    column_widths: Dict[str, float] = {}
    for column in column_sequence:
        column_width = max(
            (w.params * width_scale for w in all_weights if (w.column or "other") == column),
            default=column_min_width,
        )
        column_widths[column] = max(column_width, column_min_width)

    min_rect_width = column_min_width * 0.2
    block_inner_widths: List[float] = []
    block_rect_widths: List[List[float]] = []
    max_inner_width = column_min_width
    for block in blocks:
        widths: List[float] = []
        for weight in block["weights"]:
            rect_width = max(weight.params * width_scale, min_rect_width)
            widths.append(rect_width)
        block_rect_widths.append(widths)
        block_width = max(max(widths, default=0.0), column_min_width)
        block_inner_widths.append(block_width)
        max_inner_width = max(max_inner_width, block_width)

    total_width_units = max_inner_width + side_padding * 2

    block_heights: List[float] = []
    for block in blocks:
        if not block["weights"]:
            block_heights.append(block_padding_top + block_padding_bottom)
            continue
        height = block_padding_top + block_padding_bottom
        for idx, weight in enumerate(block["weights"]):
            rect_height = weight.bytes_per_param * height_scale
            height += rect_height
            if idx < len(block["weights"]) - 1:
                height += weight_gap_y
        block_heights.append(height)

    block_offsets: List[float] = []
    y_cursor = global_top_margin
    for height in block_heights:
        block_offsets.append(y_cursor)
        y_cursor += height + block_gap_y

    if block_heights:
        total_height_units = y_cursor - block_gap_y + global_bottom_margin
    else:
        total_height_units = global_top_margin + global_bottom_margin

    fig_width = max(8.0, total_width_units / 9000.0)
    fig_height = max(10.0, total_height_units / 400.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for block_idx, (block, block_start, block_height, inner_width) in enumerate(
        zip(blocks, block_offsets, block_heights, block_inner_widths)
    ):
        outline_left = side_padding + (max_inner_width - inner_width) / 2
        block_outline = patches.Rectangle(
            (outline_left, block_start),
            inner_width,
            block_height,
            linewidth=0.6,
            edgecolor="#3a3a3a",
            facecolor="none",
        )
        ax.add_patch(block_outline)

        y_position = block_start + block_padding_top
        block_center = outline_left + inner_width / 2
        widths = block_rect_widths[block_idx]
        for idx, weight in enumerate(block["weights"]):
            rect_width = widths[idx] if idx < len(widths) else max(
                weight.params * width_scale, min_rect_width
            )
            rect_height = weight.bytes_per_param * height_scale
            rect = patches.Rectangle(
                (
                    block_center - rect_width / 2,
                    y_position,
                ),
                rect_width,
                rect_height,
                facecolor=COLOR_MAP.get(weight.kind, COLOR_MAP["other"]),
                edgecolor="black",
                linewidth=0.4,
            )
            ax.add_patch(rect)

            label_lines = [weight.name]
            if weight.shape:
                if len(weight.shape) == 2:
                    label_lines.append(f"{weight.shape[0]}×{weight.shape[1]}")
                else:
                    dims = "×".join(str(dim) for dim in weight.shape)
                    label_lines.append(dims)
            label_text = " " + " ".join(label_lines)
            text_x = block_center + rect_width / 2 + reference_spacing
            text_y = y_position + rect_height / 2
            ax.text(
                text_x,
                text_y,
                label_text,
                ha="left",
                va="center",
                fontsize=5,
                color="black",
            )
            y_position += rect_height + weight_gap_y

    ax.set_xlim(0, total_width_units)
    ax.set_ylim(total_height_units, 0)
    ax.axis("off")

    unique_params = sum(w.params for w in all_weights if w.counts_toward_total)
    unique_bytes = sum(w.params * w.bytes_per_param for w in all_weights if w.counts_toward_total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Diagram written to {output_path} "
        f"(unique params: {unique_params:,}, {human_bytes(unique_bytes)})."
    )


def main() -> None:
    args = parse_args()
    config_path = args.config_path.expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    dtype_label = str(cfg.get("torch_dtype", "float32"))
    dtype_bytes = dtype_to_bytes(dtype_label)
    blocks = build_blocks(cfg, dtype_bytes)

    render_diagram(
        blocks=blocks,
        output_path=args.output.expanduser(),
        dtype_label=dtype_label,
    )


if __name__ == "__main__":
    main()
