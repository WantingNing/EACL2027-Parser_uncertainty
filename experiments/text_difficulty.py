import json
import math


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parser_file(parser_root, record):
    if record["dataset"] == "gcdc":
        return parser_root / "gcdc" / record["domain"] / f"{record['doc_id']}.json"
    return parser_root / "cohesentia" / "cohe" / f"{record['doc_id']}.json"


def mtld_direction(words, threshold=0.72):
    factors = 0.0
    types = set()
    count = 0
    for word in words:
        types.add(word)
        count += 1
        if len(types) / count <= threshold:
            factors += 1
            types = set()
            count = 0
    if count:
        ttr = len(types) / count
        if ttr < 1:
            factors += (1 - ttr) / (1 - threshold)
    if not factors:
        return float(len(words))
    return len(words) / factors


def mtld(words):
    if not words:
        return 0.0
    return (mtld_direction(words) + mtld_direction(words[::-1])) / 2


def dependency_depth(token, limit=200):
    # Compare indices: token.head is a new object each time.
    depth = 0
    while token.head.i != token.i and depth < limit:
        token = token.head
        depth += 1
    return depth


def text_features(doc, edu_count):
    words = [t for t in doc if not (t.is_punct or t.is_space)]
    sentences = [s for s in doc.sents if any(not (t.is_punct or t.is_space) for t in s)]
    alpha = [t.text.lower() for t in words if t.is_alpha]
    depths = [max(dependency_depth(t) for t in s) for s in sentences]
    return {
        "n_words": len(words),
        "n_sentences": len(sentences),
        "n_edus": edu_count,
        "mean_word_length": sum(len(w) for w in alpha) / len(alpha) if alpha else 0.0,
        "mtld": mtld(alpha),
        "mean_sentence_length": len(words) / len(sentences) if sentences else float(len(words)),
        "mean_dependency_depth": sum(depths) / len(depths) if depths else 0.0,
    }


def add_text_features(records, features, parser_root, spacy_model):
    import spacy

    nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])
    docs = nlp.pipe((r["text"] for r in records), batch_size=64)
    for record, doc in zip(records, docs):
        edus = json.loads(parser_file(parser_root, record).read_text(encoding="utf-8"))["edus"]
        features[record["doc_id"]].update(text_features(doc, len(edus)))


def add_surprisal(records, features, model_id, max_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    for index, record in enumerate(records, start=1):
        ids = tokenizer(record["text"], add_special_tokens=False)["input_ids"][: max_tokens - 1]
        input_ids = torch.tensor([[bos, *ids]], device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, input_ids[0, 1:], reduction="mean").item()
        features[record["doc_id"]].update({
            "lm_surprisal": nll / math.log(2),
            "lm_tokens": len(ids),
            "lm_truncated": len(ids) == max_tokens - 1,
        })
        if index % 500 == 0:
            print(f"LM surprisal: {index}/{len(records)}", flush=True)


def run(config, root):
    settings = config["text_difficulty"]
    records = read_jsonl(root / config["paths"]["llm_inputs"] / "all.jsonl")
    features = {r["doc_id"]: {"doc_id": r["doc_id"]} for r in records}
    add_text_features(records, features, root / config["paths"]["rst_parser"], settings["spacy_model"])
    add_surprisal(records, features, settings["lm"], settings["max_tokens"])
    output = root / config["paths"]["text_difficulty"]
    write_jsonl(output, [features[r["doc_id"]] for r in records])
    print(f"Wrote {output}")
