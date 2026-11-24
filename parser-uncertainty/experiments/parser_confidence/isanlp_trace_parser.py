"""Local wrappers around isanlp-rst that preserve original parsing logic.

These classes do not modify the installed ``isanlp-rst`` package. Instead,
they reuse the original predictors and intercept the pointer network outputs
at inference time so the returned ``DiscourseUnit`` tree keeps the split
probability distribution used to compute entropy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import razdel
import torch

from isanlp.annotation_rst import DiscourseUnit
from isanlp_rst.dmrst_parser.predictor import PredictorDMRST
from isanlp_rst.dmrst_parser.src.parser.data import Data
from isanlp_rst.parser import Parser as BaseParser
from isanlp_rst.universal_parser.predictor import PredictorUniRST
from isanlp_rst.utils.du_converter import DUConverter


class _PointerTraceWrapper(torch.nn.Module):
    """Record split distributions while delegating to the original pointer."""

    def __init__(self, pointer: torch.nn.Module, sink: list[list[float]]):
        super().__init__()
        self.pointer = pointer
        self.sink = sink

    def forward(self, *args: Any, **kwargs: Any):
        atten_weights, log_atten_weights = self.pointer(*args, **kwargs)
        probs = torch.exp(log_atten_weights.detach()).squeeze(0)
        self.sink.append([float(item) for item in probs.cpu().tolist()])
        return atten_weights, log_atten_weights


class _LabelTraceWrapper(torch.nn.Module):
    """Record relation-label distributions while delegating to the classifier."""

    def __init__(self, classifier: torch.nn.Module, sink: list[list[float]]):
        super().__init__()
        self.classifier = classifier
        self.sink = sink

    def forward(self, *args: Any, **kwargs: Any):
        relation_weights, log_relation_weights = self.classifier(*args, **kwargs)
        probs = relation_weights.detach().squeeze(0)
        self.sink.append([float(item) for item in probs.cpu().tolist()])
        return relation_weights, log_relation_weights


def _split_joint_label(label: str) -> tuple[str, str | None]:
    if "_" not in label:
        return label, None
    relation, nuclearity = label.rsplit("_", 1)
    if nuclearity in {"NS", "SN", "NN"}:
        return relation, nuclearity
    return label, None


def build_relation_label_set(relation_table: Sequence[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for item in relation_table:
        relation, _ = _split_joint_label(str(item))
        if relation not in seen:
            seen.add(relation)
            labels.append(relation)
    return labels


def collapse_joint_distribution(
    distribution: Sequence[float],
    relation_table: Sequence[str],
) -> tuple[list[float], list[float]]:
    relation_labels = build_relation_label_set(relation_table)
    relation_index = {label: idx for idx, label in enumerate(relation_labels)}
    nuclearity_index = {"NS": 0, "SN": 1, "NN": 2}

    relation_distribution = [0.0] * len(relation_labels)
    nuclearity_distribution = [0.0, 0.0, 0.0]

    for probability, joint_label in zip(distribution, relation_table):
        relation, nuclearity = _split_joint_label(str(joint_label))
        relation_distribution[relation_index[relation]] += float(probability)
        if nuclearity in nuclearity_index:
            nuclearity_distribution[nuclearity_index[nuclearity]] += float(probability)

    return relation_distribution, nuclearity_distribution


class TraceableDUConverter(DUConverter):
    """Keep split distributions on internal tree nodes."""

    def collect(self, tokens=None):
        data = []
        split_batches = self.predictions.get("split_distributions") or []
        relation_batches = self.predictions.get("joint_relation_distributions") or []
        relation_table = self.predictions.get("relation_table") or []

        for i in range(len(self.predictions["tokens"])):
            gold_tokens = tokens[i] if tokens else None
            edus = self._lists_to_isanlp_format(
                tokens=self.predictions["tokens"][i],
                edu_breaks=self.predictions["edu_breaks"][i],
                gold_tokens=gold_tokens,
            )
            if len(edus) == 1:
                return edus

            self.du_id = len(edus)
            split_distributions = split_batches[i] if i < len(split_batches) else []
            joint_relation_distributions = relation_batches[i] if i < len(relation_batches) else []
            rels = self._tree_string_to_list(
                self.predictions["spans"][i][0],
                split_distributions,
                joint_relation_distributions,
                relation_table,
            )
            tree = self.construct_tree(0, edus, rels)
            data.append(tree)

        return data

    @staticmethod
    def _tree_string_to_list(
        description,
        split_distributions,
        joint_relation_distributions,
        relation_table,
    ):
        rels = []
        split_index = 0
        relation_index = 0

        for rel in description.split(" "):
            if not rel:
                continue

            left, right = rel.split(",")
            left_start, left_label, left_end = left[1:].split(":")
            entropy = 0.0
            if ";entropy=" in left_label:
                left_label, entropy_str = left_label.split(";entropy=")
                try:
                    entropy = float(entropy_str)
                except ValueError:
                    entropy = 0.0

            right_start, right_label, right_end = right[:-1].split(":")
            nuclearity = left_label[0] + right_label[0]
            relation = left_label.split("=")[1] if nuclearity == "SN" else right_label.split("=")[1]

            split_candidate_count = max(int(right_end) - int(left_start), 1)
            split_distribution: list[float] | None
            if split_candidate_count == 1:
                split_distribution = [1.0]
            else:
                split_distribution = None
                if split_index < len(split_distributions):
                    candidate_distribution = [float(item) for item in split_distributions[split_index]]
                    split_index += 1
                    if len(candidate_distribution) == split_candidate_count:
                        split_distribution = candidate_distribution

            relation_distribution = None
            nuclearity_distribution = None
            if relation_index < len(joint_relation_distributions):
                joint_distribution = [
                    float(item) for item in joint_relation_distributions[relation_index]
                ]
                relation_index += 1
                if relation_table and len(joint_distribution) == len(relation_table):
                    relation_distribution, nuclearity_distribution = collapse_joint_distribution(
                        joint_distribution,
                        relation_table,
                    )

            rels.append(
                (
                    int(left_start) - 1,
                    int(left_end) - 1,
                    relation,
                    nuclearity,
                    int(right_start) - 1,
                    int(right_end) - 1,
                    entropy,
                    split_distribution,
                    relation_distribution,
                    nuclearity_distribution,
                )
            )

        return rels

    def construct_tree(self, root, edus, rels):
        (
            left_start,
            left_end,
            relation,
            nuclearity,
            right_start,
            right_end,
            entropy,
            split_distribution,
            relation_distribution,
            nuclearity_distribution,
        ) = rels[root]

        if left_start == left_end:
            left = edus[left_start]
        else:
            left_root = self._get_child(left_start, left_end, rels)
            left = self.construct_tree(left_root, edus, rels)

        if right_start == right_end:
            right = edus[right_start]
        else:
            right_root = self._get_child(right_start, right_end, rels)
            right = self.construct_tree(right_root, edus, rels)

        self.du_id += 1
        du = DiscourseUnit(
            id=self.du_id,
            left=left,
            right=right,
            entropy=entropy,
            relation=relation,
            nuclearity=nuclearity,
            start=left.start,
            end=right.end,
            text=left.text + " " + right.text,
        )
        du.split_distribution = split_distribution
        du.relation_distribution = relation_distribution
        du.nuclearity_distribution = nuclearity_distribution
        return du


class TraceablePredictorDMRST(PredictorDMRST):
    """Reuse PredictorDMRST while preserving uncertainty distributions."""

    def _run_testing_loss_with_trace(self, batch: Data, use_pred_segmentation: bool):
        split_distributions: list[list[float]] = []
        joint_relation_distributions: list[list[float]] = []
        original_pointer = self.model.pointer
        original_label_classifier = self.model.label_classifier
        self.model.pointer = _PointerTraceWrapper(original_pointer, split_distributions)
        self.model.label_classifier = _LabelTraceWrapper(
            original_label_classifier,
            joint_relation_distributions,
        )
        try:
            outputs = self.model.testing_loss(
                batch.input_sentences,
                batch.sent_breaks,
                batch.entity_ids,
                batch.entity_position_ids,
                batch.edu_breaks,
                batch.relation_label,
                batch.parsing_breaks,
                generate_tree=True,
                use_pred_segmentation=use_pred_segmentation,
            )
        finally:
            self.model.pointer = original_pointer
            self.model.label_classifier = original_label_classifier

        return outputs, split_distributions, joint_relation_distributions

    def _build_predictions(
        self,
        batch: Data,
        span_batch,
        predict_edu_breaks,
        split_distributions: list[list[float]],
        joint_relation_distributions: list[list[float]],
    ) -> dict[str, Any]:
        return {
            "tokens": [self.tokenizer.convert_ids_to_tokens(text) for text in batch.input_sentences],
            "spans": span_batch,
            "edu_breaks": predict_edu_breaks,
            "true_spans": batch.golden_metric,
            "true_edu_breaks": batch.edu_breaks,
            "split_distributions": [split_distributions],
            "joint_relation_distributions": [joint_relation_distributions],
            "relation_table": list(self.relation_table),
        }

    def parse_rst(self, text: str):
        razdel_tokens = list(razdel.tokenize(text))
        tokenized_text = [token.text for token in razdel_tokens]
        offset_positions, original_offsets = self.build_offset_converter_from_razdel(razdel_tokens)
        data = {
            "input_sentences": [tokenized_text],
            "edu_breaks": [[]],
            "decoder_input": [[]],
            "relation_label": [[]],
            "parsing_breaks": [[]],
            "golden_metric": [[]],
        }

        if len(tokenized_text) < 3:
            tree = TraceableDUConverter.dummy_tree(tokenized_text)
            self.remap_tree_offsets(tree, offset_positions, original_offsets, text)
            return {"rst": [tree]}

        batch = self.tokenize(Data(**data))
        with torch.inference_mode():
            outputs, split_distributions, joint_relation_distributions = self._run_testing_loss_with_trace(
                batch, use_pred_segmentation=True
            )

        _, _, span_batch, _, predict_edu_breaks = outputs
        predictions = self._build_predictions(
            batch=batch,
            span_batch=span_batch,
            predict_edu_breaks=predict_edu_breaks,
            split_distributions=split_distributions,
            joint_relation_distributions=joint_relation_distributions,
        )
        tree = TraceableDUConverter(predictions, tokenization_type="default").collect()[0]
        self.remap_tree_offsets(tree, offset_positions, original_offsets, text)
        return {"rst": [tree]}

    def parse_from_edus(self, edus: Sequence[str]):
        normalized_edus = self._validate_edus(edus)
        text, spans = self._compute_edu_char_spans(normalized_edus)

        razdel_tokens = list(razdel.tokenize(text))
        tokenized_text = [token.text for token in razdel_tokens]
        offset_positions, original_offsets = self.build_offset_converter_from_razdel(razdel_tokens)

        if not tokenized_text:
            raise ValueError("Unable to tokenize text derived from the provided EDUs.")

        if len(normalized_edus) == 1:
            tree = TraceableDUConverter.dummy_tree(tokenized_text)
            self.remap_tree_offsets(tree, offset_positions, original_offsets, text)
            leaves: list[str] = []
            self._collect_leaf_texts(tree, leaves)
            if leaves != normalized_edus:
                raise ValueError("Failed to align the provided EDU with the parser output.")
            return {"rst": [tree]}

        edu_breaks = self._char_spans_to_token_breaks(razdel_tokens, spans)
        num_edus = len(edu_breaks)
        relation_placeholder = [[0] * max(num_edus - 1, 0)]
        parsing_placeholder = [[0] * max(num_edus - 1, 0)]

        batch = self.tokenize(
            Data(
                input_sentences=[tokenized_text],
                edu_breaks=[edu_breaks],
                decoder_input=[[]],
                relation_label=relation_placeholder,
                parsing_breaks=parsing_placeholder,
                golden_metric=[[]],
            )
        )

        with torch.inference_mode():
            outputs, split_distributions, joint_relation_distributions = self._run_testing_loss_with_trace(
                batch, use_pred_segmentation=False
            )

        _, _, span_batch, _, predict_edu_breaks = outputs
        predictions = self._build_predictions(
            batch=batch,
            span_batch=span_batch,
            predict_edu_breaks=predict_edu_breaks,
            split_distributions=split_distributions,
            joint_relation_distributions=joint_relation_distributions,
        )
        tree = TraceableDUConverter(predictions, tokenization_type="default").collect()[0]
        self.remap_tree_offsets(tree, offset_positions, original_offsets, text)

        leaves: list[str] = []
        self._collect_leaf_texts(tree, leaves)
        if leaves != normalized_edus:
            raise ValueError("The produced segmentation does not match the provided EDUs.")

        return {"rst": [tree]}


class TraceableParser(BaseParser):
    """A drop-in Parser replacement that keeps split distributions on nodes."""

    def __init__(
        self,
        model_dir: str = None,
        hf_model_name: str = "tchewik/isanlp_rst_v3",
        hf_model_version: str = None,
        relinventory: str = None,
        relinventory_idx: int = 0,
        cuda_device: int = -1,
    ):
        if hf_model_version in self.DMRST_PARSERS:
            self.predictor = TraceablePredictorDMRST(
                model_dir=model_dir,
                hf_model_name=hf_model_name,
                hf_model_version=hf_model_version,
                cuda_device=cuda_device,
            )
        elif hf_model_version in self.UNIVERSAL_PARSERS:
            self.predictor = PredictorUniRST(
                model_dir=model_dir,
                hf_model_name=hf_model_name,
                hf_model_version=hf_model_version,
                relinventory=relinventory,
                relinventory_idx=relinventory_idx,
                cuda_device=cuda_device,
            )
        else:
            raise NotImplementedError(
                f"Available options for hf_model_version are: {', '.join(self.AVAILABLE_VERSIONS)}"
            )
