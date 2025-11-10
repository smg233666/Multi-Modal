from typing import Any, Dict, List
from PIL import Image
from vllm import LLM, SamplingParams, ModelRegistry
from transformers import AutoProcessor, AutoConfig
from jinja2 import Template
import json
import fire
import os
import re
import inspect
from tqdm import tqdm
from Assets.MATHVmain.evaluation.utils import is_equal as mathv_is_equal
from steer_qwen3_vl_vllm import SteerQwen3VLForConditionalGeneration


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")

def extract_boxed_content(text: str) -> str:
    if not isinstance(text, str) or "\\boxed{" not in text:
        return "None"

    # Fast path: simple regex match of the last non-nested pattern
    try:
        m = _BOXED_RE.findall(text)
        if m:
            return m[-1].strip()
    except Exception:
        pass

    # Balanced scan from the last occurrence of "\\boxed{"
    try:
        start = text.rfind("\\boxed{")
        if start == -1:
            return "None"
        i = start + len("\\boxed{")
        depth = 1
        j = i
        while j < len(text):
            ch = text[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[i:j].strip()
            j += 1
    except Exception:
        pass
    return "None"

def extract_choice_once_fail(text: str) -> str:
    if not isinstance(text, str):
        return "None"
    match = re.findall(
        r"(?:correct answer is|Answer[:：]?)\s*(?:\*\*)?[\(\[]?([A-E])[\)\]\.\s]?",
        text, re.IGNORECASE
    )
    if match:
        return match[-1].upper()
    match2 = re.findall(r"\b([A-E])\b", text)
    if match2:
        return match2[-1].upper()
    return "None"

def extract_numeric_fallback(text: str) -> str:
    if not isinstance(text, str):
        return "None"

    # Prefer LaTeX fraction if present
    try:
        fracs = re.findall(r"\\frac\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}", text)
        if fracs:
            a, b = fracs[-1]
            if b != '0':
                return f"{a}/{b}"
    except Exception:
        pass

    # Fall back to last integer/decimal number
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    return "None"

def mathv_equal(pred: str, gt) -> bool:
    if gt is None or pred is None:
        return False
    if isinstance(gt, list):
        return any(mathv_equal(pred, g) for g in gt)
    try:
        return mathv_is_equal(str(pred), str(gt))
    except Exception:
        return False


# ---------------------------- Main pipeline ----------------------------

def generate_and_evaluate(
    model_path="Qwen/Qwen3-VL-8B-Thinking",
    dataname="MATH-V",
    data_path="",
    base_save_path="",
    generation_save_path="",
    overall_trend_save_path="",
    batch_size=64,  # Increase the batch_size to improve throughput.
    vote_num=4,  
    tensor_parallel_size=1,  # Use tensor parallel.
    max_tokens=40960, # Match official Qwen3-VL generation length
    steering_vector_path="Empty",
    steering_strength=0.0,
    images_root="",        # base folder for images (MATH-V)
    outputs_dir="",        # path to the MATH-V repo's outputs/ (e.g., "/path/to/MATH-V/outputs")
    run_name="Qwen3VL_Thinking_8B",  # filename for predictions JSONL (outputs/<run_name>.jsonl)
):    
    if base_save_path and not os.path.exists(base_save_path):
        os.makedirs(base_save_path, exist_ok=True)

    #@ Register steering-enabled model if requested
    steering_enabled = steering_vector_path != "Empty"
    if steering_enabled:
        try:
            ModelRegistry.register_model(
                "Qwen3VLForConditionalGeneration",
                SteerQwen3VLForConditionalGeneration)
        except ValueError:
            # Already registered in this process.
            pass
        print("Registered SteerQwen3VLForConditionalGeneration for vLLM.")
    else:
        os.environ.pop("steering_vector_path", None)
        os.environ.pop("steering_strength_list", None)

    available_params = set(inspect.signature(SamplingParams).parameters)
    sampling_args = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": max_tokens,
        "n": vote_num,
        "stop": ["<|im_end|>"]
    }
    sampling_args = {k: v for k, v in sampling_args.items() if k in available_params}
    sampling_params = SamplingParams(**sampling_args)

    #@ Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    if not getattr(processor, 'apply_chat_template', None):
        raise RuntimeError('Loaded processor does not support apply_chat_template.')
    if not getattr(processor, 'chat_template', None) and getattr(tokenizer, 'chat_template', None):
        processor.chat_template = tokenizer.chat_template
    template_jinja = """\
    This is the problem:
    {{prompt}}
    """
    prompt_template = Template(template_jinja)
    
    #@ Create VLLM instance
    def create_llm(steering_strength, steering_vector_path, model_path, tensor_parallel_size=1):
        #@ Clear old environment variables
        os.environ.pop("steering_strength", None)

        #@ Set new environment variables when steering is enabled
        if steering_enabled:
            config = AutoConfig.from_pretrained(model_path)
            steering_strength_list = [steering_strength] * getattr(config, "num_hidden_layers", 1)
            print(f"Set steering_strength_list to: {steering_strength_list}")
            os.environ["steering_strength_list"] = ",".join(map(str, steering_strength_list))

            os.environ["steering_vector_path"] = steering_vector_path
            print(f"Set steering_vector_path to: {steering_vector_path}")
        else:
            os.environ.pop("steering_strength_list", None)
            os.environ.pop("steering_vector_path", None)
        
        #@ Force garbage collection, Avoid Heisenbug
        import gc
        gc.collect()
        
        #@ Create new LLM instance
        return LLM(model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                max_model_len=max_tokens,
                gpu_memory_utilization=0.9,
                limit_mm_per_prompt={"image": 8})

    llm = create_llm(steering_strength, steering_vector_path, model_path, tensor_parallel_size)
    
    #@ Load data (supports .json and .jsonl)
    if data_path.endswith(".jsonl"):
        question_dataset = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                question_dataset.append(json.loads(line))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            question_dataset = json.load(f)
    
    #@ Keep a copy of the original data, ensure the output format is consistent
    original_data = question_dataset.copy()
    
    #@ Preprocess data
    print(f"Preprocessing {len(question_dataset)} datapoints")
    processed_prompts = []
    images_batch = []  # aligned with prompts; each item is list[PIL.Image] or None

    def _resolve_image_paths(dp: Dict[str, Any]) -> List[str]:
        imgs: List[str] = []
        if 'images' in dp and isinstance(dp['images'], list):
            imgs = dp['images']
        elif 'image' in dp:
            if isinstance(dp['image'], list):
                imgs = dp['image']
            elif isinstance(dp['image'], str):
                imgs = [dp['image']]

        if images_root:
            base_dir = images_root
        elif data_path:
            base_dir = os.path.dirname(data_path)
        else:
            base_dir = os.getcwd()

        resolved: List[str] = []
        for p in imgs:
            if not p:
                continue
            # Absolute path
            if os.path.isabs(p) and os.path.exists(p):
                resolved.append(p)
                continue

            base_name = os.path.basename(p)
            cand1 = os.path.join(base_dir, base_name)
            if os.path.exists(cand1):
                resolved.append(cand1)
                continue

            cand2 = os.path.join(base_dir, p)
            if os.path.exists(cand2):
                resolved.append(cand2)
                continue
        return resolved

    # Track basic image load stats
    zero_image_qids = []

    for datapoint in tqdm(question_dataset):
        # Prefer 'problem' (MATH) but fallback to 'question' (MATH-V)
        problem = datapoint.get('problem') or datapoint.get('question') or ""

        # Some datasets include literal placeholders like "<image1>" in the text. These are not required for Qwen's multimodal input and can confuse models.
        problem = re.sub(r"<image\d+>", "", problem, flags=re.IGNORECASE)

        prompt_temp = prompt_template.render(prompt=problem)

        # Collect PIL images and stable UUIDs based on source paths
        pil_images: List[Image.Image] = []
        for img_path in _resolve_image_paths(datapoint):
            try:
                im = Image.open(img_path).convert("RGB")
                pil_images.append(im)
            except Exception as err:
                print(f"[warn] failed to load image {img_path}: {err}")
                continue
        if not pil_images:
            qid = datapoint.get('question_id') or datapoint.get('id') or datapoint.get('uid')
            print(f"[warn] no image loaded for question {qid}")
            if qid is not None:
                zero_image_qids.append(qid)

        # Build messages with explicit image entries so the processor renders correct placeholders
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": ([{"type": "image"}] * len(pil_images)) + [{"type": "text", "text": prompt_temp}]},
        ]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Quick sanity: the number of placeholders should equal number of images
        placeholder_count = prompt_text.count("<|image_pad|>")
        if placeholder_count != len(pil_images):
            qid = datapoint.get('question_id') or datapoint.get('id') or datapoint.get('uid')
            print(f"[warn] placeholder/image mismatch for question {qid}: placeholders={placeholder_count}, images={len(pil_images)}")
        processed_prompts.append(prompt_text)
        images_batch.append(pil_images if pil_images else None)

    print('len(processed_prompts):', len(processed_prompts))
    if zero_image_qids:
        print(f"[summary] {len(zero_image_qids)} samples had no images attached. Example IDs: {zero_image_qids[:5]}")

    # #@ Generate texts (batched for throughput, order preserved)
    structured_inputs = []
    for p_text, imgs in zip(processed_prompts, images_batch):
        if imgs:
            structured_inputs.append({"prompt": p_text, "multi_modal_data": {"image": imgs}})
        else:
            structured_inputs.append({"prompt": p_text})

    all_generated_texts: List[List[str]] = []
    total = len(structured_inputs)
    for start in range(0, total, batch_size):
        batch = structured_inputs[start:start + batch_size]
        outputs = llm.generate(prompts=batch, sampling_params=sampling_params)
        # outputs aligns with batch order
        for req_out in outputs:
            texts = [o.text for o in req_out.outputs]
            all_generated_texts.append(texts)

        done = min(start + batch_size, total)
        if done % max(10, batch_size) == 0 or done == total:
            print(f"Processed {done}/{total} samples (batched)")
    
    #@ Add generated texts back to original data, keep the original format
    results_for_saving = []
    for i in range(len(all_generated_texts)):
        datapoint = original_data[i]
        #@ Ground truth candidates (letter and mapped option text when available)
        gt_candidates = []
        if 'answer' in datapoint:
            raw_ans = datapoint['answer']
            # normalize to list of string candidates
            if isinstance(raw_ans, list):
                gt_candidates.extend([str(a) for a in raw_ans if a is not None])
            elif raw_ans is not None:
                gt_candidates.append(str(raw_ans))

            # If multiple-choice options exist and answer is a letter (A–E), also include the corresponding option text as an acceptable target.
            opts = datapoint.get('options')
            if isinstance(raw_ans, str) and isinstance(opts, list) and opts:
                letter = raw_ans.strip().upper()
                if len(letter) == 1 and 'A' <= letter <= 'E':
                    idx = ord(letter) - ord('A')
                    if 0 <= idx < len(opts):
                        opt_val = opts[idx]
                        if opt_val is not None:
                            gt_candidates.append(str(opt_val))
        
        #@ If there are generated texts, add them to the data point
        one_record = datapoint.copy()
        # Stable question id for evaluator output
        qid = datapoint.get('question_id') or datapoint.get('id') or datapoint.get('uid') or f"idx_{i}"
        one_record['question_id'] = qid
        one_record['llm_reasoning'], one_record['llm_answer'], one_record['llm_final_answer'], one_record['is_correct'] = [], [], [], []
        one_record['llm_reasoning_token_num'], one_record['llm_answer_token_num'] = [], []
        
        one_generation = all_generated_texts[i]
        for rollout in one_generation:
            # Split around closing </think>; treat content after it as visible answer
            think_idx = rollout.find('</think>')
            if think_idx != -1:
                llm_reasoning = rollout[:think_idx]
                post_think = rollout[think_idx + len('</think>'):]
                llm_answer = post_think.strip()
            else:
                llm_reasoning = rollout
                post_think = rollout
                llm_answer = rollout.strip()

            opts = datapoint.get('options')
            has_options = isinstance(opts, list) and len(opts) > 0

            # First, prefer boxed answer from the visible part
            llm_final_answer = extract_boxed_content(post_think)
            if llm_final_answer == "None":
                if has_options:
                    # Prefer choice first
                    llm_final_answer = extract_choice_once_fail(post_think)
                    # Allow numeric only if it matches an option text
                    if llm_final_answer == "None":
                        cand = extract_numeric_fallback(post_think)
                        if cand != "None" and mathv_equal(cand, opts):
                            llm_final_answer = cand
                else:
                    llm_final_answer = extract_numeric_fallback(post_think)
            # As last resort, try the whole generated text
            if llm_final_answer == "None":
                llm_final_answer = extract_boxed_content(rollout)
            if llm_final_answer == "None":
                if has_options:
                    llm_final_answer = extract_choice_once_fail(rollout)
                    if llm_final_answer == "None":
                        cand2 = extract_numeric_fallback(rollout)
                        if cand2 != "None" and mathv_equal(cand2, opts):
                            llm_final_answer = cand2
                else:
                    llm_final_answer = extract_numeric_fallback(rollout)

            # Token counts
            llm_reasoning_token_num = len(tokenizer.encode(llm_reasoning))
            llm_answer_token_num = len(tokenizer.encode(llm_answer))

            is_correct = mathv_equal(llm_final_answer, gt_candidates)
            
            one_record['llm_reasoning'].append(llm_reasoning)
            one_record['llm_answer'].append(llm_answer)
            one_record['llm_final_answer'].append(llm_final_answer)
            one_record['is_correct'].append(is_correct)
            
            one_record['llm_reasoning_token_num'].append(llm_reasoning_token_num)
            one_record['llm_answer_token_num'].append(llm_answer_token_num)
                
        # Averages for dashboarding
        def _avg(xs): 
            return sum(xs)/len(xs) if xs else 0.0
        one_record['avg_llm_reasoning_token_num'] = _avg(one_record['llm_reasoning_token_num'])
        one_record['avg_llm_answer_token_num'] = _avg(one_record['llm_answer_token_num'])
        one_record['accuracy'] = _avg(one_record['is_correct'])
        
        results_for_saving.append(one_record)
    
    #@ Write evaluator-ready predictions directly into MATH-V repo's outputs/
    eval_out = os.path.join(outputs_dir, f"{run_name}.jsonl")
    os.makedirs(os.path.dirname(eval_out), exist_ok=True)
    with open(eval_out, "w", encoding="utf-8") as wf:
        for rec in results_for_saving:
            fa_list = rec.get('llm_final_answer', []) or ["None"]
            majority = max(set(fa_list), key=fa_list.count) if fa_list else "None"
            wf.write(json.dumps({
                "question_id": rec.get("question_id"),
                "prediction": majority
            }, ensure_ascii=False) + "\n")
    print(f"Wrote evaluator file to: {eval_out}")
    
    #@ Save verbose results
    with open(generation_save_path, 'w', encoding='utf-8') as f:
        json.dump(results_for_saving, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {generation_save_path}")
        
    total_accuracy = sum([datapoint['accuracy'] for datapoint in results_for_saving]) / len(results_for_saving) if results_for_saving else 0.0
    print(f"Total accuracy: {total_accuracy}")
    
    #@ Calculate average length
    def _avg_all(key):
        return sum([dp.get(key, 0.0) for dp in results_for_saving]) / len(results_for_saving) if results_for_saving else 0.0
    total_llm_reasoning_token_num = _avg_all('avg_llm_reasoning_token_num')
    total_llm_answer_token_num = _avg_all('avg_llm_answer_token_num')
    print(f"Average reasoning token num: {total_llm_reasoning_token_num}")
    print(f"Average answer token num: {total_llm_answer_token_num}")
    
    #@ Save overall trend data (now also includes avg_think tokens)
    if os.path.exists(overall_trend_save_path) and os.path.getsize(overall_trend_save_path) > 0:
        with open(overall_trend_save_path, 'r', encoding='utf-8') as f:
            overall_trend_data = json.load(f)
    else:
        overall_trend_data = []

    new_entry = {
        "strength": steering_strength,
        "total_accuracy": total_accuracy,
        "average_reasoning_token_num": total_llm_reasoning_token_num,
        "average_answer_token_num": total_llm_answer_token_num,
    }

    existing_entry = next((entry for entry in overall_trend_data if entry["strength"] == steering_strength), None)
    if existing_entry:
        existing_entry.update(new_entry)
    else:
        overall_trend_data.append(new_entry)

    with open(overall_trend_save_path, 'w', encoding='utf-8') as f:
        json.dump(overall_trend_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    fire.Fire(generate_and_evaluate)
