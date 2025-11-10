import argparse
import json
import os
import os.path as op
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from jinja2 import Template
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoTokenizer
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        val = v.lower()
        if val in {"yes", "true", "t", "y", "1"}:
            return True
        if val in {"no", "false", "f", "n", "0"}:
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed text-image pairs with Qwen3-VL")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Thinking",
        help="Path or HF repo id for the Qwen3-VL model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/Questions/MathV.jsonl",
        help="Path to input data (.json or .jsonl).",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="Data/Images/MathV",
        help="Directory containing image assets referenced by the dataset.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="Data/Representation/MathV",
        help="Directory to store the resulting embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of samples to process per batch when all share the same modality.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length passed to the tokenizer.",
    )
    parser.add_argument(
        "--reasoning",
        type=str2bool,
        default=True,
        help="If true, add reasoning instructions to the prompt.",
    )
    return parser.parse_args()


def load_dataset(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: str) -> None:
    if not op.exists(path):
        os.makedirs(path, exist_ok=True)


def resolve_image_paths(
    datapoint: Dict[str, Any],
    images_root: str,
) -> List[str]:
    """Resolve dataset image references to concrete filesystem paths."""
    entries: Sequence[str] = ()
    if "images" in datapoint and isinstance(datapoint["images"], list):
        entries = datapoint["images"]
    elif "image" in datapoint:
        image_val = datapoint["image"]
        if isinstance(image_val, list):
            entries = image_val
        elif isinstance(image_val, str):
            entries = [image_val]

    resolved: List[str] = []
    for entry in entries:
        if not entry:
            continue
        if op.isabs(entry) and op.exists(entry):
            resolved.append(entry)
            continue
        candidate = op.join(images_root, op.basename(entry))
        if op.exists(candidate):
            resolved.append(candidate)
    return resolved


def load_images(image_paths: Sequence[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[warn] failed to load image {path}: {exc}")
    return images


def build_prompt_text(
    datapoint: Dict[str, Any],
    template: Template,
) -> str:
    problem = datapoint.get("problem") or datapoint.get("question") or ""
    return template.render(prompt=problem)


def gather_prompts_and_images(
    records: Sequence[Dict[str, Any]],
    template: Template,
    images_root: str,
) -> Tuple[List[str], List[List[str]], List[str]]:
    prompts: List[str] = []
    image_paths: List[List[str]] = []
    missing_images: List[str] = []
    for idx, record in enumerate(records):
        prompt_text = build_prompt_text(record, template)
        resolved_paths = resolve_image_paths(record, images_root)
        if not resolved_paths:
            qid = record.get("question_id") or record.get("id") or record.get("uid") or str(idx)
            missing_images.append(qid)
        prompts.append(prompt_text)
        image_paths.append(resolved_paths)
    return prompts, image_paths, missing_images


def make_messages(
    user_text: str,
    num_images: int,
    reasoning: bool,
) -> List[Dict[str, Any]]:
    system_prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}."
        if reasoning
        else None
    )
    user_content: List[Any] = []
    for _ in range(num_images):
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": user_text})

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def forward_batch(
    prompts: Sequence[str],
    image_paths_list: Sequence[Sequence[str]],
    processor: Qwen3VLProcessor,
    model: Qwen3VLForConditionalGeneration,
    reasoning: bool,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    if not prompts:
        return tuple()

    # Always process per-sample to avoid placeholder/token mismatches.
    per_sample_outputs: List[Tuple[torch.Tensor, ...]] = []
    for prompt, paths in zip(prompts, image_paths_list):
        images = load_images(paths)
        messages = make_messages(prompt, len(images), reasoning)
        prompt_str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs = {
            "text": prompt_str,
            "return_tensors": "pt",
            "padding": True,
            # Do NOT use truncation here; it can desync image placeholders vs features.
        }
        if images:
            proc_kwargs["images"] = images
        inputs = processor(**proc_kwargs)
        inputs = move_to_device(inputs, device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # Keep only last-token representations per layer to ensure consistent shapes
        per_sample_outputs.append(
            tuple(layer[:, -1, :].detach().cpu().to(torch.float32) for layer in out.hidden_states)
        )

    if not per_sample_outputs:
        return tuple()
    num_layers = len(per_sample_outputs[0])
    stacked: List[torch.Tensor] = []
    for layer_idx in range(num_layers):
        stacked.append(torch.cat([s[layer_idx] for s in per_sample_outputs], dim=0))
    return tuple(stacked)


def accumulate_hidden_states(
    accumulator: List[np.ndarray],
    hidden_states: Tuple[torch.Tensor, ...],
) -> List[np.ndarray]:
    if not hidden_states:
        return accumulator

    seq_embeds: List[np.ndarray] = []
    for layer in hidden_states:
        arr = layer.detach().cpu().to(torch.float32).numpy()
        if arr.ndim == 3:
            # [B, T, H] -> take last token: [B, H]
            arr = arr[:, -1, :]
        elif arr.ndim == 2:
            # [B, H] already last-token
            pass
        else:
            raise RuntimeError(f"Unexpected hidden state rank {arr.ndim}; expected 2 or 3.")
        seq_embeds.append(arr)
    if not accumulator:
        return [embed.copy() for embed in seq_embeds]

    if len(accumulator) != len(seq_embeds):
        raise RuntimeError("Layer count mismatch while aggregating hidden states.")

    for idx, layer_embed in enumerate(seq_embeds):
        accumulator[idx] = np.concatenate((accumulator[idx], layer_embed), axis=0)
    return accumulator


def main() -> None:
    args = parse_args()
    model_name = args.model_path.split("/")[-1]
    save_dir = op.join(args.save_path, f"{model_name}{'' if args.reasoning else '_no_reasoning'}")
    ensure_dir(save_dir)

    records = load_dataset(args.data_path)

    template_str = (
        "Please reason step by step, and put your final answer within \\boxed{}.\n{{prompt}}"
        if args.reasoning
        else "{{prompt}}"
    )
    prompt_template = Template(template_str)

    prompts, image_path_groups, missing_images = gather_prompts_and_images(
        records, prompt_template, args.images_root
    )
    if missing_images:
        print(f"[info] {len(missing_images)} prompts missing images; falling back to text-only prompts.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    video_processor = Qwen3VLVideoProcessor()
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        video_processor=video_processor,
    )
    if not getattr(processor, "chat_template", None):
        processor.chat_template = tokenizer.chat_template

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Move fully to a single device (ensures GPU use without Accelerate)
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    print(f"Model device: {next(model.parameters()).device}")

    accumulator: List[np.ndarray] = []
    batch_size = max(1, args.batch_size)
    model_device = next(model.parameters()).device
    for start in tqdm(range(0, len(prompts), batch_size)):
        end = start + batch_size
        batch_prompts = prompts[start:end]
        batch_images = image_path_groups[start:end]
        hidden_states = forward_batch(
            batch_prompts,
            batch_images,
            processor,
            model,
            args.reasoning,
            args.max_length,
            model_device,
        )
        accumulator = accumulate_hidden_states(accumulator, hidden_states)

    for layer_idx, embeddings in enumerate(accumulator):
        np.save(op.join(save_dir, f"embeds_{layer_idx}.npy"), embeddings)

    print("finished")


if __name__ == "__main__":
    main()
