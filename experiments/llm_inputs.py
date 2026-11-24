import json
import math
import random
from collections import defaultdict

GCDC_DOMAINS = ("clinton", "enron", "yahoo", "yelp")
REFERENCE_KEYS = ("human_label_3", "label_3", "raw_score_5", "score", "score_source", "split_origin")
PROMPT_FIELDS = ("rst_relation_entropy", "rst_top1", "rst_margin")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_5_to_3(score):
    if score <= 2:
        return 1
    if score <= 4:
        return 2
    return 3


def gcdc_records(gcdc_root, domain):
    records = []
    for row in read_jsonl(gcdc_root / f"gcdc_{domain}" / f"gcdc_{domain}_all.jsonl"):
        label = int(row["score"])
        records.append({
            "dataset": "gcdc",
            "domain": domain,
            "doc_id": f"{domain}_{row['text_id']}",
            "text_id": str(row["text_id"]),
            "text": row["text"],
            "human_label_3": label,
            "label_3": label,
        })
    return records


def cohesentia_records(cohesentia_root):
    records = []
    for row in read_jsonl(cohesentia_root / "cohe" / "cohe_all.jsonl"):
        records.append({
            "dataset": "cohesentia",
            "domain": "cohe",
            "doc_id": row["doc_id"],
            "text_id": str(row["text_id"]),
            "title": row["title"],
            "text": row["text"],
            "raw_score_5": int(row["raw_score_5"]),
            "label_3": score_5_to_3(int(row["raw_score_5"])),
            "score": row["score"],
            "score_source": row["score_source"],
            "split_origin": row["split_origin"],
        })
    return records


def relation_stats(distribution):
    values = [max(float(x), 0.0) for x in distribution]
    total = sum(values)
    probs = [x / total for x in values]
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    ranked = sorted(probs, reverse=True)
    return entropy / math.log(len(probs)), ranked[0], ranked[0] - ranked[1]


def parser_metrics(parser_path):
    """Document-level means over the internal nodes of the RST tree."""
    with open(parser_path, encoding="utf-8") as handle:
        nodes = json.load(handle)["tree"]["nodes"]
    entropies, raw_entropies, top1s, margins = [], [], [], []
    for node in nodes:
        if node["is_leaf"]:
            continue
        entropy, top1, margin = relation_stats(node["relation_distribution"])
        entropies.append(entropy)
        top1s.append(top1)
        margins.append(margin)
        raw_entropies.append(float(node["relation_entropy"]))
    return {
        "rst_n_internal_nodes": len(entropies),
        "rst_relation_entropy": sum(entropies) / len(entropies),
        "rst_relation_entropy_raw": sum(raw_entropies) / len(raw_entropies),
        "rst_top1": sum(top1s) / len(top1s),
        "rst_margin": sum(margins) / len(margins),
    }


def parser_path(parser_root, record):
    if record["dataset"] == "gcdc":
        return parser_root / "gcdc" / record["domain"] / f"{record['doc_id']}.json"
    return parser_root / "cohesentia" / "cohe" / f"{record['doc_id']}.json"


def split_record(record, metrics, path):
    sample = {key: value for key, value in record.items() if key not in REFERENCE_KEYS}
    sample.update(metrics)
    sample["parser_path"] = path
    reference = {"doc_id": record["doc_id"], "dataset": record["dataset"], "domain": record["domain"]}
    for key in REFERENCE_KEYS:
        if key in record:
            reference[key] = record[key]
    return sample, reference


def mismatched_permutation(values, rng):
    """A permutation in which no document receives its own (or an identical) confidence triple."""
    n = len(values)
    order = list(range(n))
    # Sattolo's algorithm: one cycle, so no document is mapped to itself.
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        order[i], order[j] = order[j], order[i]
    for _ in range(100 * n):
        clashes = [i for i in range(n) if values[order[i]] == values[i]]
        if not clashes:
            return order
        i = clashes[0]
        j = rng.randrange(n)
        if values[order[j]] != values[i] and values[order[i]] != values[j]:
            order[i], order[j] = order[j], order[i]
    raise RuntimeError("Could not build a mismatched permutation")


def shuffle_confidence(samples, seed):
    """Within each dataset/domain, show every document the confidence triple of another document."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[(sample["dataset"], sample["domain"])].append(index)
    shuffled = [dict(sample) for sample in samples]
    for key in sorted(groups):
        indices = groups[key]
        values = [tuple(samples[i][field] for field in PROMPT_FIELDS) for i in indices]
        order = mismatched_permutation(values, rng)
        for position, source in zip(indices, order):
            donor = samples[indices[source]]
            for field in PROMPT_FIELDS:
                shuffled[position][f"prompt_{field}"] = donor[field]
            shuffled[position]["prompt_rst_source_doc_id"] = donor["doc_id"]
            shuffled[position]["shuffle_seed"] = seed
    return shuffled


def run(config, root):
    paths = config["paths"]
    parser_root = root / paths["rst_parser"]
    records = []
    for domain in GCDC_DOMAINS:
        records.extend(gcdc_records(root / paths["gcdc"], domain))
    records.extend(cohesentia_records(root / paths["cohesentia"]))

    samples, references = [], []
    for record in records:
        path = parser_path(parser_root, record)
        sample, reference = split_record(record, parser_metrics(path), str(path.relative_to(root)))
        samples.append(sample)
        references.append(reference)

    input_dir = root / paths["llm_inputs"]
    write_jsonl(input_dir / "all.jsonl", samples)
    write_jsonl(input_dir / "all_shuffled_confidence.jsonl", shuffle_confidence(samples, config["llm"]["shuffle_seed"]))
    write_jsonl(root / paths["llm_reference"] / "reference_labels.jsonl", references)
    print(f"Wrote {len(samples)} documents to {input_dir}")
