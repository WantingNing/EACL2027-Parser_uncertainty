# When Discourse Parsers Are Uncertain

Code for *When Discourse Parsers Are Uncertain: Parser Confidence as a Diagnostic Signal for Coherence Assessment* (EACL 2027).

We study whether discourse-parser confidence tracks human coherence judgments, test it as an input to a
coherence classifier (ReCo) and in ensembles with existing models, and compare it with zero-shot LLM judgments.

## Repository

```
config.yaml                      all parameters and run recipes
experiments/
  run.py                         entry point: python experiments/run.py <recipe>
  ensemble.py                    probability-interpolation ensembles
  text_difficulty.py             text-difficulty features (length, lexical, syntactic, LM surprisal)
  llm_inputs.py                  LLM inputs, including the shuffled-confidence control
  llm_inference.py               zero-shot LLM judgments
  prompts/                       prompt templates
models/
  reco/                          ReCo trainer, also used for the text-only baseline and the ablations
  entyrelcoh/, relcoh/           baselines EntyRelCoh and RelCoh (Liu and Strube, 2025a, 2025b)
parser-uncertainty/experiments/  RST parsing with IsaNLP-RST, keeping the relation posteriors
```

## Setup

Python 3.10 with PyTorch, Transformers, scikit-learn, NumPy, SciPy and PyYAML.

Only needed for some steps:

- RST parsing: `isanlp_rst`, `isanlp`, `razdel`
- text-difficulty features: `spacy` with `en_core_web_sm`
- local LLMs: `accelerate`; GPT models: `openai` and `OPENAI_API_KEY`
- RelCoh: [GloVe](https://nlp.stanford.edu/projects/glove/) 840B 300d at `assets/glove.840B.300d.txt` (or set `paths.glove` in `config.yaml`)

## Data

The datasets are not redistributed here. Obtain [GCDC](https://github.com/aylai/GCDC-corpus) (Lai and Tetreault, 2018) from its authors and
CoheSentia (Maimon and Tsarfaty, 2023), and place them as follows (paths are set in `config.yaml`):

```
data/
  GCDC/
    raw/gcdc/{Clinton,Enron,Yahoo,Yelp}_{train,test}.csv      original GCDC files
    gcdc_<domain>/gcdc_<domain>_all.jsonl                     all documents: text_id, text, score
    gcdc_<domain>/<fold 1-10>/{train,dev,test}.json           folds of Liu and Strube (2025a, 2025b)
  CoheSentia/
    cohe/cohe_all.jsonl                                       all stories with raw_score_5 and score_id
    cohe/<fold 1-5>/{train,dev,test}.json                     official test set, five train/dev folds
    raw/cohe/<fold 1-5>/{train,dev,test}.json
  prepared/{text_only,entyrelcoh,relcoh}/<dataset>/<fold>/{train,dev,test}.json   model inputs
```

CoheSentia's five-point scores are mapped to three classes: 1-2 Low, 3-4 Medium, 5 High.

## Released outputs

The outputs used in the paper. The parser and LLM outputs contain the document text, so they are on
[Hugging Face](https://huggingface.co/datasets/WillaNing/EACL2027-Parser_uncertainty-outputs) with access granted
to those who already have GCDC.

| File | Download | Unpack into | Content |
|---|---|---|---|
| `parsers.tar.gz` | Hugging Face | `parser-uncertainty/data/parsers/` | RST-DT and GUM-RRG parses ([IsaNLP-RST](https://github.com/tchewik/isanlp_rst) 3.2.0), PDTB parses ([DisCoPy](https://github.com/rknaebel/discopy)) |
| `llm_predictions.tar.gz` | Hugging Face | `results/llm/predictions/` | responses of the five LLMs for every prompt variant |
| `text_difficulty_features.jsonl` | [release](../../releases) | `results/parser_confidence/` | text-difficulty features |

To parse the documents yourself (GPU recommended):

```bash
python parser-uncertainty/experiments/parser_confidence/get_parser_info.py --source gcdc --gcdc-root data/GCDC \
    --parser-type rstdt --output-root parser-uncertainty/data/parsers/rst --cuda-device 0
python parser-uncertainty/experiments/parser_confidence/get_parser_info.py --source jsonl \
    --input-jsonl data/CoheSentia/cohe/cohe_all.jsonl --source-name cohesentia --group-name cohe \
    --parser-type rstdt --output-root parser-uncertainty/data/parsers/rst --cuda-device 0
```

Use `--parser-type gumrrg` for the GUM-RRG parser.

## Running

All settings are in `config.yaml`. Each recipe is one command, run from the repository root:

```bash
python experiments/run.py <recipe>
```

| Recipe | What it does | Paper |
|---|---|---|
| `cache` | builds graph, text and RST-tensor caches; run before training | |
| `main` | trains ReCo, Text, EntyRelCoh and RelCoh on all datasets | Table 2 |
| `ablations` | trains the ReCo component ablations on CoheSentia | Table 22 |
| `ensembles` | equal and dev-weighted ensembles of all model pairs | Tables 2, 16-21 |
| `text_difficulty` | computes text-difficulty features | Tables 8, 9 |
| `llm_inputs` | builds LLM inputs and the shuffled-confidence inputs | |
| `llm` | LLM judgments with the default, RST-confidence and self-confidence prompts | Section 5, Tables 13, 15 |
| `llm_shuffled` | LLM judgments with shuffled parser confidence | Table 14 |

Training uses seed 106524, one process per fold, and selects checkpoints on dev accuracy. Ensemble
weights are always selected on the dev split. Outputs go to `runs/`:

```
runs/train/seed_<seed>/<dataset>/<model>/...            per-fold predictions and metrics
runs/ensembles/seed_<seed>/<dataset>/<pair>/<mode>/     ensemble predictions and summary.json
runs/llm/predictions/<model>[_<prompt>].jsonl           LLM responses
```

A new experiment is a new entry under `variants`, `ensemble.pairs` or `recipes` in `config.yaml`.

## References

- Alice Lai and Joel Tetreault. 2018. [Discourse Coherence in the Wild: A Dataset, Evaluation and Methods](https://aclanthology.org/W18-5023/). SIGDIAL.
- Aviya Maimon and Reut Tsarfaty. 2023. [CoheSentia: A Novel Benchmark of Incremental versus Holistic Assessment of Coherence in Generated Texts](https://aclanthology.org/2023.emnlp-main.324/). EMNLP.
- Wei Liu and Michael Strube. 2025a. [Discourse Relation-Enhanced Neural Coherence Modeling](https://aclanthology.org/2025.acl-long.236/). ACL. (RelCoh)
- Wei Liu and Michael Strube. 2025b. [Joint Modeling of Entities and Discourse Relations for Coherence Assessment](https://aclanthology.org/2025.emnlp-main.1113/). EMNLP. (EntyRelCoh)
