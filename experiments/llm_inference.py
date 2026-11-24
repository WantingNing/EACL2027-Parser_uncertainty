import json
import re

LABEL_RE = re.compile(r"Coherence\s*:\s*([1-5])", re.IGNORECASE)
CONFIDENCE_RE = re.compile(r"Confidence\s*:\s*(\d{1,3}(?:\.\d+)?)\s*%?", re.IGNORECASE)
CONFIDENCE_LINE_RE = re.compile(r"Confidence\s*:[^\n]*", re.IGNORECASE)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_label(text, max_label):
    match = LABEL_RE.search(text)
    if match:
        value = int(match.group(1))
        return value if value <= max_label else None
    # Without "Coherence:", take the first standalone digit outside the confidence line.
    for token in re.findall(r"\b[1-5]\b", CONFIDENCE_LINE_RE.sub("", text)):
        if int(token) <= max_label:
            return int(token)
    return None


def parse_confidence(text):
    match = CONFIDENCE_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    return value if value <= 100 else None


def score_5_to_3(score):
    if score <= 2:
        return 1
    if score <= 4:
        return 2
    return 3


def build_prompt(template, record, variant):
    # The shuffled-confidence inputs show another document's values, stored as prompt_rst_*.
    prefix = "prompt_" if variant == "shuffled_confidence" else ""
    return template.format(
        text=record["text"],
        title=record.get("title", ""),  # only CoheSentia stories have a title
        rst_n_internal_nodes=record["rst_n_internal_nodes"],
        rst_relation_entropy=record[prefix + "rst_relation_entropy"],
        rst_top1=record[prefix + "rst_top1"],
        rst_margin=record[prefix + "rst_margin"],
    )


def load_local_model(model_id, dtype_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype_name), device_map="auto", trust_remote_code=True)
    return tokenizer, model


def local_generate(tokenizer, model, prompt, generation):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]
    temperature = float(generation["temperature"])
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(generation["max_new_tokens"]),
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
        top_p=float(generation["top_p"]),
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()


def openai_generate(client, model_id, prompt, generation):
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(generation["temperature"]),
        max_tokens=int(generation["max_new_tokens"]),
    )
    # Aliases such as "gpt-4o" resolve to dated snapshots; keep the one that answered.
    return (response.choices[0].message.content or "").strip(), response.model


def prediction_row(record, model_key, model_cfg, variant, raw_response, served_model):
    label = parse_label(raw_response, 3 if record["dataset"] == "gcdc" else 5)
    row = {**record, "model_key": model_key, "model_id": model_cfg["model_id"],
           "raw_response": raw_response, "parse_ok": label is not None}
    if record["dataset"] == "gcdc":
        row["llm_label_3"] = label
    else:
        row["llm_label_5"] = label
        row["llm_label_3"] = score_5_to_3(label) if label is not None else None
    if variant == "self_confidence":
        confidence = parse_confidence(raw_response)
        row["llm_confidence"] = confidence
        row["confidence_parse_ok"] = confidence is not None
    row["prompt_variant"] = variant
    if served_model:
        row["served_model"] = served_model
    return row


def run(config, root, model_key, variant):
    """One model and one prompt variant over all documents; rerunning resumes the parsed rows."""
    llm = config["llm"]
    model_cfg = llm["models"][model_key]
    generation = dict(llm["generation"])
    if variant == "self_confidence":
        generation["max_new_tokens"] = llm["self_confidence_max_new_tokens"]

    input_name = "all_shuffled_confidence.jsonl" if variant == "shuffled_confidence" else "all.jsonl"
    records = read_jsonl(root / config["paths"]["llm_inputs"] / input_name)
    templates = {}
    for dataset in ("gcdc", "cohesentia"):
        templates[dataset] = (root / config["paths"]["prompts"] / f"{dataset}_{variant}.txt").read_text(encoding="utf-8")
    suffix = "" if variant == "default" else f"_{variant}"
    output = root / config["paths"]["runs"] / "llm" / "predictions" / f"{model_key}{suffix}.jsonl"
    done = {}
    if output.exists():
        for row in read_jsonl(output):
            if row["parse_ok"]:
                done[row["doc_id"]] = row

    if model_cfg["provider"] == "hf":
        tokenizer, model = load_local_model(model_cfg["model_id"], llm["hf_dtype"])
    else:
        from openai import OpenAI
        client = OpenAI(max_retries=6)  # reads OPENAI_API_KEY

    rows = []
    for index, record in enumerate(records, start=1):
        if record["doc_id"] in done:
            rows.append(done[record["doc_id"]])
            continue
        prompt = build_prompt(templates[record["dataset"]], record, variant)
        served_model = None
        try:
            if model_cfg["provider"] == "hf":
                raw_response = local_generate(tokenizer, model, prompt, generation)
            else:
                raw_response, served_model = openai_generate(client, model_cfg["model_id"], prompt, generation)
        except Exception as error:  # keep long runs going; failed rows are retried on the next run
            print(f"{model_key} {record['doc_id']}: {type(error).__name__}: {error}", flush=True)
            raw_response = ""
        rows.append(prediction_row(record, model_key, model_cfg, variant, raw_response, served_model))
        if index % 50 == 0:
            write_jsonl(output, rows)
            print(f"{model_key} {variant}: {index}/{len(records)}", flush=True)
    write_jsonl(output, rows)
    print(f"Wrote {output}")
