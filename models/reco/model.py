from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


NUCLEARITY_TO_ID = {"<LEAF>": 0, "NS": 1, "SN": 2, "NN": 3}
NUC_PATH_TO_ID = {"none": 0, "N": 1, "S": 2, "NN": 3}
RELATION_DIST_SIZE = 18
TOPOLOGY_DIM = 5
DOC_PROFILE_DIM = 26
NODE_ENTROPY_SOURCES = {"relation", "nuclearity", "split"}
PARSER_CONFIDENCE_MODES = {"original", "zero", "uniform", "shuffle"}
PARSER_PROFILE_MODES = {"all", "confidence", "size"}


def _load_sota_base_classifier_cls():
    repo_root = Path(__file__).resolve().parents[2]
    fusion_dir = repo_root / "models" / "entyrelcoh"
    module_path = fusion_dir / "model.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find SOTA fusion model.py: {module_path}")
    if str(fusion_dir) not in sys.path:
        sys.path.insert(0, str(fusion_dir))
    spec = importlib.util.spec_from_file_location("entyrelcoh_fusion_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import SOTA fusion model from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BaseClassifer


SOTA_BASE_CLASSIFIER_CLS = _load_sota_base_classifier_cls()


class SharedDocumentEncoder(nn.Module):
    def __init__(
        self,
        hf_model_name: str,
        freeze_encoder: bool = False,
        text_pooling: str = "mean",
        use_precomputed_text_features: bool = False,
    ):
        super().__init__()
        if text_pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported text_pooling: {text_pooling}")
        self.use_precomputed_text_features = bool(use_precomputed_text_features)
        if self.use_precomputed_text_features:
            config = AutoConfig.from_pretrained(hf_model_name)
            self.encoder = None
            self.hidden_size = int(config.hidden_size)
        else:
            self.encoder = AutoModel.from_pretrained(hf_model_name)
            self.hidden_size = int(self.encoder.config.hidden_size)
        self.freeze_encoder = bool(freeze_encoder)
        self.text_pooling = text_pooling
        if self.encoder is not None and self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder is not None and self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.encoder is None:
            raise RuntimeError("The encoder is disabled because precomputed text features are enabled.")
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    def _pool_document(self, sequence_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.text_pooling == "cls":
            return sequence_states[:, 0]
        weights = attention_mask.to(dtype=sequence_states.dtype).unsqueeze(-1)
        summed = (sequence_states * weights).sum(dim=1)
        return summed / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        window_input_ids: torch.Tensor,
        window_attention_mask: torch.Tensor,
        window_to_global: torch.Tensor,
        encode_windows: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.encoder is None:
            raise RuntimeError("Precomputed text features must be supplied by the batch.")
        doc_outputs = self._encode(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
        doc_repr = self._pool_document(doc_outputs.last_hidden_state, doc_attention_mask)
        if not encode_windows:
            return None, doc_repr

        batch_size, num_windows, window_length = window_input_ids.shape
        flat_ids = window_input_ids.reshape(batch_size * num_windows, window_length)
        flat_mask = window_attention_mask.reshape(batch_size * num_windows, window_length)
        encoded = self._encode(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        encoded = encoded.reshape(batch_size, num_windows, window_length, self.hidden_size)

        has_global = bool((window_to_global >= 0).any())
        max_global = int(window_to_global.max().item()) + 1 if has_global else 1
        global_states = encoded.new_zeros(batch_size, max_global, self.hidden_size)
        global_weights = encoded.new_zeros(batch_size, max_global, 1)
        for batch_index in range(batch_size):
            for window_index in range(num_windows):
                mapping = window_to_global[batch_index, window_index]
                valid = mapping >= 0
                if not bool(valid.any()):
                    continue
                indices = mapping[valid]
                states = encoded[batch_index, window_index, valid]
                num_valid = int(valid.sum().item())
                local_positions = torch.arange(num_valid, device=states.device, dtype=states.dtype)
                weights = torch.minimum(local_positions + 1.0, torch.flip(local_positions, dims=[0]) + 1.0)
                weights = weights.unsqueeze(-1)
                global_states[batch_index].index_add_(0, indices, states * weights)
                global_weights[batch_index].index_add_(0, indices, weights)

        global_states = global_states / global_weights.clamp_min(1.0)
        return global_states, doc_repr


class MeanSpanPooler(nn.Module):
    def forward(self, sequence_states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        valid = token_indices[(token_indices >= 0) & (token_indices < sequence_states.size(0))]
        if valid.numel() == 0:
            return sequence_states.new_zeros(sequence_states.size(-1))
        return sequence_states[valid].mean(dim=0)


class SotaStyleClassificationHead(nn.Module):
    """Thin adapter around EntyRelCoh/fusion/model.py::BaseClassifer."""

    def __init__(self, input_size: int, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        args = SimpleNamespace(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_labels=int(num_labels),
            dropout=float(dropout),
        )
        self.base = SOTA_BASE_CLASSIFIER_CLS(args)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Use the SOTA forward path. In Eval mode BaseClassifer returns
        # `(preds, logits)`, and our training loss applies CE to those logits.
        outputs = self.base(features, flag="Eval")
        if isinstance(outputs, tuple) and len(outputs) > 1:
            return outputs[1]

        # Some older copies of EntyRelCoh return only predictions in Eval mode.
        # Keep the same BaseClassifer layers but compute logits directly.
        if features.dtype == torch.bfloat16:
            features = features.float()
        x = self.base.dropout(features)
        x = x.float()
        x = self.base.fc1(x)
        x = self.base.dropout(x)
        x = self.base.fc2(x)
        x = self.base.dropout(x)
        return self.base.classifier(x)


class TreeFormerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        bias_dim: int,
        tree_dist_cap: int,
        ffn_multiplier: int,
        dropout: float,
        use_distance_bias: bool = True,
        use_ancestor_bias: bool = True,
        use_subtree_bias: bool = True,
        use_source_relation_bias: bool = True,
        use_relation_uncertainty_bias: bool = True,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_distance_bias = use_distance_bias
        self.use_ancestor_bias = use_ancestor_bias
        self.use_subtree_bias = use_subtree_bias
        self.use_source_relation_bias = use_source_relation_bias
        self.use_relation_uncertainty_bias = use_relation_uncertainty_bias

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dist_embedding = nn.Embedding(tree_dist_cap + 1, bias_dim)
        self.ancestor_embedding = nn.Embedding(3, bias_dim)
        self.subtree_embedding = nn.Embedding(2, bias_dim)
        self.source_relation_projection = nn.Linear(bias_dim, num_heads, bias=False)
        self.dist_projection = nn.Linear(bias_dim, num_heads, bias=False)
        self.ancestor_projection = nn.Linear(bias_dim, num_heads, bias=False)
        self.subtree_projection = nn.Linear(bias_dim, num_heads, bias=False)
        self.rel_uncertainty_projection = nn.Linear(5, num_heads, bias=False)

        ffn_hidden = hidden_size * ffn_multiplier
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
        )

    def _pair_bias(
        self,
        dist_bucket: torch.Tensor,
        ancestor_relation: torch.Tensor,
        same_subtree: torch.Tensor,
        source_relation_bias_input: torch.Tensor,
        parser_stats: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = dist_bucket.size(0)
        bias = source_relation_bias_input.new_zeros(self.num_heads, num_nodes, num_nodes)
        if self.use_distance_bias:
            dist_bias = self.dist_projection(self.dist_embedding(dist_bucket)).permute(2, 0, 1)
            bias = bias + dist_bias
        if self.use_ancestor_bias:
            ancestor_bias = self.ancestor_projection(self.ancestor_embedding(ancestor_relation + 1)).permute(2, 0, 1)
            bias = bias + ancestor_bias
        if self.use_subtree_bias:
            subtree_bias = self.subtree_projection(self.subtree_embedding(same_subtree)).permute(2, 0, 1)
            bias = bias + subtree_bias
        if self.use_source_relation_bias:
            source_bias = self.source_relation_projection(source_relation_bias_input).transpose(0, 1)
            bias = bias + source_bias.unsqueeze(1).expand(-1, num_nodes, -1)
        if self.use_relation_uncertainty_bias:
            uncertainty_bias = self.rel_uncertainty_projection(parser_stats).transpose(0, 1)
            bias = bias + uncertainty_bias.unsqueeze(1).expand(-1, num_nodes, -1)
        return bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        dist_bucket: torch.Tensor,
        ancestor_relation: torch.Tensor,
        same_subtree: torch.Tensor,
        source_relation_bias_input: torch.Tensor,
        parser_stats: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = hidden_states.size(0)
        q = self.q_proj(hidden_states).view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(hidden_states).view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(hidden_states).view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + self._pair_bias(
            dist_bucket=dist_bucket,
            ancestor_relation=ancestor_relation,
            same_subtree=same_subtree,
            source_relation_bias_input=source_relation_bias_input,
            parser_stats=parser_stats,
        )
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(weights), v).transpose(0, 1).reshape(num_nodes, self.hidden_size)
        hidden_states = self.attn_norm(hidden_states + self.dropout(self.out_proj(attended)))
        return self.ffn_norm(hidden_states + self.dropout(self.ffn(hidden_states)))


class TreeFormerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        bias_dim: int,
        tree_dist_cap: int,
        ffn_multiplier: int,
        dropout: float,
        use_distance_bias: bool,
        use_ancestor_bias: bool,
        use_subtree_bias: bool,
        use_source_relation_bias: bool,
        use_relation_uncertainty_bias: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TreeFormerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    bias_dim=bias_dim,
                    tree_dist_cap=tree_dist_cap,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                    use_distance_bias=use_distance_bias,
                    use_ancestor_bias=use_ancestor_bias,
                    use_subtree_bias=use_subtree_bias,
                    use_source_relation_bias=use_source_relation_bias,
                    use_relation_uncertainty_bias=use_relation_uncertainty_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout_score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        root_index: int,
        dist_bucket: torch.Tensor,
        ancestor_relation: torch.Tensor,
        same_subtree: torch.Tensor,
        source_relation_bias_input: torch.Tensor,
        parser_stats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                dist_bucket=dist_bucket,
                ancestor_relation=ancestor_relation,
                same_subtree=same_subtree,
                source_relation_bias_input=source_relation_bias_input,
                parser_stats=parser_stats,
            )
        weights = torch.softmax(self.readout_score(hidden_states).squeeze(-1), dim=0)
        pooled = (weights.unsqueeze(-1) * hidden_states).sum(dim=0)
        return torch.cat([pooled, hidden_states[root_index]], dim=-1), weights


class ReCo(nn.Module):
    def __init__(
        self,
        hf_model_name: str,
        num_labels: int,
        relation_vocab: dict[str, int],
        hidden_size: int = 256,
        dropout: float = 0.2,
        relation_embedding_dim: int = 32,
        nuclearity_embedding_dim: int = 16,
        path_embedding_dim: int = 32,
        text_hidden_dim: int = 256,
        treeformer_layers: int = 2,
        treeformer_heads: int = 4,
        treeformer_bias_dim: int = 32,
        treeformer_ffn_multiplier: int = 4,
        tree_dist_cap: int = 8,
        max_path_depth: int = 64,
        relation_label_dropout: float = 0.2,
        relation_dist_dropout: float = 0.1,
        use_text_branch: bool = True,
        use_treeformer_branch: bool = True,
        use_comp_branch: bool = True,
        use_relation_distribution: bool = True,
        use_relation_embedding: bool = True,
        use_parser_dropout: bool = True,
        use_distance_bias: bool = True,
        use_ancestor_bias: bool = True,
        use_subtree_bias: bool = True,
        use_source_relation_bias: bool = True,
        use_relation_uncertainty_bias: bool = True,
        unordered_tree: bool = False,
        node_entropy_mode: str = "original",
        node_entropy_source: str = "relation",
        parser_confidence_mode: str = "original",
        use_doc_entropy_features: bool = True,
        use_node_non_entropy_parser_stats: bool = True,
        use_span_states: bool = True,
        use_nuclearity_embedding: bool = True,
        use_path_features: bool = True,
        use_topology_features: bool = True,
        use_doc_relation_histogram: bool = True,
        use_parser_profile: bool = True,
        parser_profile_mode: str = "all",
        freeze_encoder: bool = False,
        text_pooling: str = "mean",
        use_precomputed_text_features: bool = False,
        classifier_hidden_size: int = 256,
    ):
        super().__init__()
        if node_entropy_mode not in {"original", "zero", "shuffle"}:
            raise ValueError(f"Unsupported node_entropy_mode: {node_entropy_mode}")
        if node_entropy_source not in NODE_ENTROPY_SOURCES:
            raise ValueError(f"Unsupported node_entropy_source: {node_entropy_source}")
        if parser_confidence_mode not in PARSER_CONFIDENCE_MODES:
            raise ValueError(f"Unsupported parser_confidence_mode: {parser_confidence_mode}")
        if parser_profile_mode not in PARSER_PROFILE_MODES:
            raise ValueError(f"Unsupported parser_profile_mode: {parser_profile_mode}")
        self.num_labels = num_labels
        self.relation_vocab = relation_vocab
        self.use_text_branch = use_text_branch
        self.use_treeformer_branch = use_treeformer_branch
        self.use_comp_branch = use_comp_branch
        self.use_parser_profile = bool(use_parser_profile)
        self.needs_node_structure = bool(self.use_treeformer_branch or self.use_comp_branch)
        self.needs_structure = bool(self.needs_node_structure or self.use_parser_profile)
        self.use_relation_distribution = use_relation_distribution
        self.use_relation_embedding = use_relation_embedding
        self.use_parser_dropout = use_parser_dropout
        self.relation_dist_dropout = float(relation_dist_dropout)
        self.tree_dist_cap = int(tree_dist_cap)
        self.max_path_depth = int(max_path_depth)
        self.node_entropy_mode = node_entropy_mode
        self.node_entropy_source = node_entropy_source
        self.parser_confidence_mode = parser_confidence_mode
        self.use_doc_entropy_features = bool(use_doc_entropy_features)
        self.use_node_non_entropy_parser_stats = bool(use_node_non_entropy_parser_stats)
        self.use_span_states = bool(use_span_states)
        self.use_nuclearity_embedding = bool(use_nuclearity_embedding)
        self.use_path_features = bool(use_path_features)
        self.use_topology_features = bool(use_topology_features)
        self.use_doc_relation_histogram = bool(use_doc_relation_histogram)
        self.parser_profile_mode = parser_profile_mode

        self.document_encoder = SharedDocumentEncoder(
            hf_model_name,
            freeze_encoder=freeze_encoder,
            text_pooling=text_pooling,
            use_precomputed_text_features=use_precomputed_text_features,
        )
        self.encoder_hidden_size = int(self.document_encoder.hidden_size)
        self.span_pooler = MeanSpanPooler()
        self.structural_size = (3 * hidden_size) + DOC_PROFILE_DIM
        if not self.needs_structure:
            # In text-only mode, construct the SOTA BaseClassifer before any
            # RST modules consume RNG.  This keeps the initialization and input
            # dimensionality aligned with EntyRelCoh's own text-only model.
            self.text_projection = nn.Identity()
            self.ce_head = SotaStyleClassificationHead(
                input_size=self.encoder_hidden_size,
                hidden_size=classifier_hidden_size,
                num_labels=num_labels,
                dropout=dropout,
            )
            return

        text_only_head_init = SotaStyleClassificationHead(
            input_size=self.encoder_hidden_size,
            hidden_size=classifier_hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )

        self.text_projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, text_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_hidden_dim, hidden_size),
        )
        self.relation_embedding = nn.Embedding(len(relation_vocab), relation_embedding_dim)
        self.source_relation_embedding = nn.Embedding(len(relation_vocab) + 1, treeformer_bias_dim)
        self.leaf_relation_id = len(relation_vocab)
        self.nuclearity_embedding = nn.Embedding(len(NUCLEARITY_TO_ID), nuclearity_embedding_dim)
        self.depth_path_embedding = nn.Embedding(max_path_depth + 1, path_embedding_dim)
        self.branch_path_embedding = nn.Embedding(2, path_embedding_dim)
        self.nuc_path_embedding = nn.Embedding(len(NUC_PATH_TO_ID), path_embedding_dim)
        self.relation_label_dropout = nn.Dropout(relation_label_dropout)

        self.parser_observable_size = RELATION_DIST_SIZE + 5 + relation_embedding_dim
        node_input_size = (
            self.encoder_hidden_size
            + self.parser_observable_size
            + nuclearity_embedding_dim
            + TOPOLOGY_DIM
            + path_embedding_dim
        )
        self.node_input_projection = nn.Linear(node_input_size, hidden_size)
        self.treeformer = TreeFormerEncoder(
            hidden_size=hidden_size,
            num_layers=treeformer_layers,
            num_heads=treeformer_heads,
            bias_dim=treeformer_bias_dim,
            tree_dist_cap=tree_dist_cap,
            ffn_multiplier=treeformer_ffn_multiplier,
            dropout=dropout,
            use_distance_bias=use_distance_bias,
            use_ancestor_bias=use_ancestor_bias,
            use_subtree_bias=use_subtree_bias,
            use_source_relation_bias=use_source_relation_bias,
            use_relation_uncertainty_bias=use_relation_uncertainty_bias,
        )
        if self.use_comp_branch:
            raise ValueError("Ordered composition has been removed from ReCo. Set use_comp_branch=false.")

        final_size = (
            self.encoder_hidden_size + self.structural_size
            if self.needs_structure
            else self.encoder_hidden_size
        )
        self.ce_head = SotaStyleClassificationHead(
            input_size=final_size,
            hidden_size=classifier_hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )
        self._warm_start_full_head_from_text_only(text_only_head_init)

    def _warm_start_full_head_from_text_only(self, text_only_head: SotaStyleClassificationHead) -> None:
        """Initialize full head as exact text-only head plus zero RST columns.

        This makes the full model initially equivalent to the SOTA text-only
        classifier: the first 768 columns of fc1 reuse the text-only weights,
        the newly added RST-feature columns are zero, and the downstream MLP
        layers are copied exactly.  Training can then learn an RST increment
        without first damaging the strong text-only baseline.
        """
        with torch.no_grad():
            full = self.ce_head.base
            text = text_only_head.base
            full.fc1.weight.zero_()
            full.fc1.weight[:, : self.encoder_hidden_size].copy_(text.fc1.weight)
            full.fc1.bias.copy_(text.fc1.bias)
            full.fc2.weight.copy_(text.fc2.weight)
            full.fc2.bias.copy_(text.fc2.bias)
            full.classifier.weight.copy_(text.classifier.weight)
            full.classifier.bias.copy_(text.classifier.bias)

    def encoder_parameters(self):
        return [param for param in self.document_encoder.parameters() if param.requires_grad]

    def head_parameters(self):
        head_ids = {id(param) for param in self.ce_head.parameters()}
        return [param for param in self.parameters() if id(param) in head_ids]

    def other_parameters(self):
        encoder_ids = {id(param) for param in self.document_encoder.parameters()}
        head_ids = {id(param) for param in self.head_parameters()}
        return [param for param in self.parameters() if id(param) not in encoder_ids and id(param) not in head_ids]

    def relation_id(self, label: str | None) -> int:
        if not self.use_relation_embedding or label is None:
            return 0
        return self.relation_vocab.get(str(label), 0)

    def nuclearity_id(self, label: str | None) -> int:
        return NUCLEARITY_TO_ID.get(str(label), NUCLEARITY_TO_ID["NN"])

    def _maybe_span_state(self, span_state: torch.Tensor) -> torch.Tensor:
        return span_state if self.use_span_states else torch.zeros_like(span_state)

    def _maybe_nuclearity_embedding(self, nuc_id: torch.Tensor) -> torch.Tensor:
        nuc_vec = self.nuclearity_embedding(nuc_id)
        return nuc_vec if self.use_nuclearity_embedding else torch.zeros_like(nuc_vec)

    def _maybe_path_vector(self, path_vector: torch.Tensor) -> torch.Tensor:
        return path_vector if self.use_path_features else torch.zeros_like(path_vector)

    def _span_indices(
        self,
        node: dict[str, Any],
        edu_to_subwords: torch.Tensor,
        edu_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        indices: list[int] = []
        for edu_offset in range(int(node["edu_start"]), int(node["edu_end"]) + 1):
            valid = edu_to_subwords[edu_offset][edu_mask[edu_offset]]
            indices.extend(valid.detach().cpu().tolist())
        if not indices:
            indices = [0]
        return torch.tensor(indices, device=device, dtype=torch.long)

    def _relation_distribution(self, node: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values = [float(value) for value in node.get("relation_distribution", [])]
        if len(values) < RELATION_DIST_SIZE:
            values = values + [0.0] * (RELATION_DIST_SIZE - len(values))
        values = values[:RELATION_DIST_SIZE]
        if not self.use_relation_distribution:
            values = [0.0] * RELATION_DIST_SIZE
        tensor = torch.tensor(values, device=device, dtype=dtype)
        if self.training and self.use_parser_dropout and self.relation_dist_dropout > 0.0:
            tensor = F.dropout(tensor, p=self.relation_dist_dropout, training=True)
        return tensor

    def _parser_stats(
        self,
        node: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        stats_override: tuple[float, float, float, float, float] | None = None,
    ) -> torch.Tensor:
        if stats_override is None:
            entropy, confidence, margin, ambiguity, rri = self._node_parser_stats_values(node)
        else:
            entropy, confidence, margin, ambiguity, rri = [float(value) for value in stats_override]
        if not self.use_node_non_entropy_parser_stats:
            confidence = 0.0
            margin = 0.0
            ambiguity = 0.0
        return torch.tensor([entropy, confidence, margin, ambiguity, rri], device=device, dtype=dtype)

    def _parser_observable(
        self,
        node: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        stats_override: tuple[float, float, float, float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation_dist = self._relation_distribution(node, device=device, dtype=dtype)
        stats = self._parser_stats(
            node,
            device=device,
            dtype=dtype,
            stats_override=stats_override,
        )
        rel_id = torch.tensor(self.relation_id(node.get("relation_label")), device=device, dtype=torch.long)
        rel_emb = self.relation_embedding(rel_id)
        if self.training and self.use_parser_dropout:
            rel_emb = self.relation_label_dropout(rel_emb)
        if not self.use_relation_embedding:
            rel_emb = torch.zeros_like(rel_emb)
        return torch.cat([relation_dist, stats, rel_emb], dim=-1), stats

    def _leaf_parser_observable(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(RELATION_DIST_SIZE + 5, device=device, dtype=dtype)
        rel_emb = torch.zeros(self.relation_embedding.embedding_dim, device=device, dtype=dtype)
        return torch.cat([zeros, rel_emb], dim=-1), torch.zeros(5, device=device, dtype=dtype)

    def _node_entropy_value(self, node: dict[str, Any]) -> float:
        key = f"{self.node_entropy_source}_entropy_norm"
        return float(node.get(key, node.get("entropy_norm", 0.0)))

    def _node_rri_value(self, node: dict[str, Any]) -> float:
        key = f"{self.node_entropy_source}_rri"
        return float(node.get(key, node.get("rri", 0.0)))

    def _node_parser_stats_values(self, node: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            self._node_entropy_value(node),
            float(node.get("confidence", 0.0)),
            float(node.get("margin", 0.0)),
            float(node.get("ambiguity", 0.0)),
            self._node_rri_value(node),
        )

    def _uniform_parser_stats_values(self) -> tuple[float, float, float, float, float]:
        uniform_top1 = 1.0 / float(RELATION_DIST_SIZE)
        uniform_top2_mass = 2.0 / float(RELATION_DIST_SIZE)
        return (1.0, uniform_top1, 0.0, uniform_top2_mass, 0.0)

    def _node_span_states(
        self,
        graph: dict[str, Any],
        global_states: torch.Tensor,
        edu_to_subwords: torch.Tensor,
        edu_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        device = global_states.device
        edu_states = []
        for edu_index in range(int(graph["num_edus"])):
            pseudo_node = {"edu_start": edu_index, "edu_end": edu_index}
            indices = self._span_indices(pseudo_node, edu_to_subwords, edu_mask, device=device)
            edu_states.append(self.span_pooler(global_states, indices))
        edu_tensor = torch.stack(edu_states, dim=0)

        span_states: list[torch.Tensor] = []
        for node in graph["nodes"]:
            indices = self._span_indices(node, edu_to_subwords, edu_mask, device=device)
            span_states.append(self.span_pooler(global_states, indices))
        return span_states, edu_tensor

    def _edge_nuc_role(self, parent_node: dict[str, Any], side: str) -> str:
        label = str(parent_node.get("nuclearity_label", "") or "")
        if label == "NS":
            return "N" if side == "left" else "S"
        if label == "SN":
            return "S" if side == "left" else "N"
        if label == "NN":
            return "NN"
        return "none"

    def _tree_metadata(
        self,
        graph: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
        nodes = graph["nodes"]
        num_nodes = len(nodes)
        key_to_index = {(str(node["kind"]), int(node["parser_id"])): int(node["index"]) for node in nodes}
        parent = [-1] * num_nodes
        branch_from_parent = [0] * num_nodes
        nuc_from_parent = [0] * num_nodes
        for parent_idx in graph["internal_indices"]:
            parent_node = nodes[parent_idx]
            for side_name, child_field, branch_id in (("left", "left_child", 0), ("right", "right_child", 1)):
                child = parent_node[child_field]
                child_idx = key_to_index[(str(child["kind"]), int(child["id"]))]
                parent[child_idx] = int(parent_idx)
                branch_from_parent[child_idx] = branch_id
                nuc_from_parent[child_idx] = NUC_PATH_TO_ID[self._edge_nuc_role(parent_node, side_name)]

        ancestor_chains: list[list[int]] = []
        path_steps: list[list[tuple[int, int]]] = []
        for node_idx in range(num_nodes):
            chain = []
            steps: list[tuple[int, int]] = []
            current = node_idx
            while current >= 0:
                chain.append(current)
                if parent[current] >= 0:
                    steps.append((branch_from_parent[current], nuc_from_parent[current]))
                current = parent[current]
            ancestor_chains.append(chain)
            path_steps.append(list(reversed(steps)))

        depths = [int(node["depth"]) for node in nodes]
        dist = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.long)
        ancestor_relation = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.long)
        same_subtree = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.long)
        ancestor_sets = [set(chain) for chain in ancestor_chains]

        root_index = int(graph["root_index"])
        root_child = [-1] * num_nodes
        for node_idx in range(num_nodes):
            if node_idx == root_index:
                continue
            current = node_idx
            while parent[current] >= 0 and parent[current] != root_index:
                current = parent[current]
            root_child[node_idx] = current if parent[current] == root_index else -1

        for i in range(num_nodes):
            for j in range(num_nodes):
                common = ancestor_sets[i].intersection(ancestor_sets[j])
                lca = max(common, key=lambda idx: depths[idx]) if common else root_index
                dist[i, j] = min(max(depths[i] + depths[j] - 2 * depths[lca], 0), self.tree_dist_cap)
                if i in ancestor_sets[j] and i != j:
                    ancestor_relation[i, j] = 1
                elif j in ancestor_sets[i] and i != j:
                    ancestor_relation[i, j] = -1
                if i == j or (root_child[i] >= 0 and root_child[i] == root_child[j]):
                    same_subtree[i, j] = 1
        return dist, ancestor_relation, same_subtree, path_steps

    def _path_vectors_from_ids(
        self,
        branch_paths: list[list[int]],
        nuc_paths: list[list[int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        vectors = []
        for branch_steps, nuc_steps in zip(branch_paths, nuc_paths):
            path_vec = torch.zeros(self.depth_path_embedding.embedding_dim, device=device, dtype=dtype)
            for one_based_depth, (branch_id, nuc_id) in enumerate(zip(branch_steps, nuc_steps), start=1):
                depth_id = min(one_based_depth, self.max_path_depth)
                path_vec = (
                    path_vec
                    + self.depth_path_embedding(torch.tensor(depth_id, device=device, dtype=torch.long))
                    + self.branch_path_embedding(torch.tensor(int(branch_id), device=device, dtype=torch.long))
                    + self.nuc_path_embedding(torch.tensor(int(nuc_id), device=device, dtype=torch.long))
                )
            vectors.append(path_vec)
        return torch.stack(vectors, dim=0)

    def _path_vectors(self, path_steps: list[list[tuple[int, int]]], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        vectors = []
        for steps in path_steps:
            path_vec = torch.zeros(self.depth_path_embedding.embedding_dim, device=device, dtype=dtype)
            for one_based_depth, (branch_id, nuc_id) in enumerate(steps, start=1):
                depth_id = min(one_based_depth, self.max_path_depth)
                path_vec = (
                    path_vec
                    + self.depth_path_embedding(torch.tensor(depth_id, device=device, dtype=torch.long))
                    + self.branch_path_embedding(torch.tensor(branch_id, device=device, dtype=torch.long))
                    + self.nuc_path_embedding(torch.tensor(nuc_id, device=device, dtype=torch.long))
                )
            vectors.append(path_vec)
        return torch.stack(vectors, dim=0)

    def _doc_profile(
        self,
        graph: dict[str, Any],
        dtype: torch.dtype,
        device: torch.device,
        node_entropy_mode: str | None = None,
    ) -> torch.Tensor:
        internal_nodes = [graph["nodes"][idx] for idx in graph["internal_indices"]]
        if not internal_nodes:
            return torch.zeros(DOC_PROFILE_DIM, device=device, dtype=dtype)
        stats_overrides = self._node_parser_stats_overrides(graph, node_entropy_mode=node_entropy_mode)
        stats_by_index = {
            int(node["index"]): stats_overrides.get(
                int(node["index"]),
                self._node_parser_stats_values(node),
            )
            for node in internal_nodes
        }
        entropies = torch.tensor([stats_by_index[int(node["index"])][0] for node in internal_nodes], device=device, dtype=dtype)
        confidences = torch.tensor([stats_by_index[int(node["index"])][1] for node in internal_nodes], device=device, dtype=dtype)
        margins = torch.tensor([stats_by_index[int(node["index"])][2] for node in internal_nodes], device=device, dtype=dtype)
        root_entropy = 0.0
        for node in internal_nodes:
            if int(node["parser_id"]) == int(graph["root_node_id"]):
                root_entropy = stats_by_index[int(node["index"])][0]
                break
        hist = torch.zeros(RELATION_DIST_SIZE, device=device, dtype=dtype)
        for node in internal_nodes:
            rel_id = self.relation_vocab.get(str(node.get("relation_label")), 0) - 1
            if 0 <= rel_id < RELATION_DIST_SIZE:
                hist[rel_id] += 1.0
        hist = hist / hist.sum().clamp_min(1.0)
        if not self.use_doc_relation_histogram:
            hist = torch.zeros_like(hist)
        scalar_profile = torch.tensor(
            [
                float(entropies.mean().item()),
                float(entropies.max().item()),
                root_entropy,
                float(confidences.mean().item()),
                float(margins.mean().item()),
                float(len(internal_nodes)),
                float(graph["num_edus"]),
                float(graph.get("max_depth", 0)),
            ],
            device=device,
            dtype=dtype,
        )
        if not self.use_doc_entropy_features:
            scalar_profile = scalar_profile.clone()
            scalar_profile[:3] = 0.0
        profile = torch.cat([scalar_profile[:5], hist, scalar_profile[5:]], dim=-1)
        if self.parser_profile_mode == "all":
            return profile
        masked_profile = torch.zeros_like(profile)
        if self.parser_profile_mode == "confidence":
            masked_profile[:5] = profile[:5]
        elif self.parser_profile_mode == "size":
            masked_profile[-3:] = profile[-3:]
        return masked_profile

    def _node_parser_stats_overrides(
        self,
        graph: dict[str, Any],
        node_entropy_mode: str | None = None,
    ) -> dict[int, tuple[float, float, float, float, float]]:
        mode = self.node_entropy_mode if node_entropy_mode is None else node_entropy_mode
        if mode not in {"original", "zero", "shuffle"}:
            raise ValueError(f"Unsupported node_entropy_mode: {mode}")
        internal_indices = [int(idx) for idx in graph["internal_indices"]]
        if self.parser_confidence_mode == "zero":
            return {idx: (0.0, 0.0, 0.0, 0.0, 0.0) for idx in internal_indices}
        if self.parser_confidence_mode == "uniform":
            uniform = self._uniform_parser_stats_values()
            return {idx: uniform for idx in internal_indices}
        if self.parser_confidence_mode == "shuffle":
            original = [self._node_parser_stats_values(graph["nodes"][idx]) for idx in internal_indices]
            if len(original) <= 1:
                shuffled = original
            else:
                digest = hashlib.md5(str(graph.get("doc_id", "")).encode("utf-8")).hexdigest()
                shift = int(digest[:8], 16) % len(original)
                if shift == 0:
                    shift = 1
                shuffled = original[shift:] + original[:shift]
            return {idx: value for idx, value in zip(internal_indices, shuffled)}

        if mode == "original":
            return {}
        if mode == "zero":
            return {
                idx: (
                    0.0,
                    float(graph["nodes"][idx].get("confidence", 0.0)),
                    float(graph["nodes"][idx].get("margin", 0.0)),
                    float(graph["nodes"][idx].get("ambiguity", 0.0)),
                    0.0,
                )
                for idx in internal_indices
            }

        original = [
            (
                self._node_entropy_value(graph["nodes"][idx]),
                self._node_rri_value(graph["nodes"][idx]),
            )
            for idx in internal_indices
        ]
        if len(original) <= 1:
            shuffled = original
        else:
            digest = hashlib.md5(str(graph.get("doc_id", "")).encode("utf-8")).hexdigest()
            shift = int(digest[:8], 16) % len(original)
            if shift == 0:
                shift = 1
            shuffled = original[shift:] + original[:shift]
        overrides = {}
        for idx, (entropy, rri) in zip(internal_indices, shuffled):
            node = graph["nodes"][idx]
            overrides[idx] = (
                entropy,
                float(node.get("confidence", 0.0)),
                float(node.get("margin", 0.0)),
                float(node.get("ambiguity", 0.0)),
                rri,
            )
        return overrides

    def _build_structural_inputs(
        self,
        graph: dict[str, Any],
        global_states: torch.Tensor,
        edu_to_subwords: torch.Tensor,
        edu_mask: torch.Tensor,
        dtype: torch.dtype,
        node_entropy_mode: str | None = None,
    ) -> dict[str, Any]:
        span_states, edu_states = self._node_span_states(graph, global_states, edu_to_subwords, edu_mask)
        dist_bucket, ancestor_relation, same_subtree, path_steps = self._tree_metadata(graph, device=global_states.device)
        path_vectors = self._path_vectors(path_steps, device=global_states.device, dtype=dtype)
        node_inputs = []
        parser_vectors: dict[int, torch.Tensor] = {}
        nuclearity_vectors: dict[int, torch.Tensor] = {}
        parser_stats_rows = []
        source_relation_ids = []
        parser_eu_values = []
        stats_overrides = self._node_parser_stats_overrides(graph, node_entropy_mode=node_entropy_mode)

        for node in graph["nodes"]:
            if node["kind"] == "internal":
                parser_vec, stats = self._parser_observable(
                    node,
                    device=global_states.device,
                    dtype=dtype,
                    stats_override=stats_overrides.get(int(node["index"])),
                )
                nuc_id = torch.tensor(self.nuclearity_id(node.get("nuclearity_label")), device=global_states.device, dtype=torch.long)
                source_rel_id = self.relation_id(node.get("relation_label"))
                parser_vectors[int(node["parser_id"])] = parser_vec
                nuclearity_vectors[int(node["parser_id"])] = self._maybe_nuclearity_embedding(nuc_id)
                parser_eu_values.append(stats[4])
            else:
                parser_vec, stats = self._leaf_parser_observable(device=global_states.device, dtype=dtype)
                nuc_id = torch.tensor(NUCLEARITY_TO_ID["<LEAF>"], device=global_states.device, dtype=torch.long)
                source_rel_id = self.leaf_relation_id
            topology = torch.tensor(node["topology"], device=global_states.device, dtype=dtype)
            if not self.use_topology_features:
                topology = torch.zeros_like(topology)
            node_inputs.append(
                torch.cat(
                    [
                        self._maybe_span_state(span_states[int(node["index"])]),
                        parser_vec,
                        self._maybe_nuclearity_embedding(nuc_id),
                        topology,
                        self._maybe_path_vector(path_vectors[int(node["index"])]),
                    ],
                    dim=-1,
                )
            )
            parser_stats_rows.append(stats)
            source_relation_ids.append(source_rel_id)

        source_relation_ids_tensor = torch.tensor(source_relation_ids, device=global_states.device, dtype=torch.long)
        if not self.use_span_states:
            edu_states = torch.zeros_like(edu_states)
        return {
            "node_hidden": self.node_input_projection(torch.stack(node_inputs, dim=0)),
            "edu_states": edu_states,
            "parser_vectors": parser_vectors,
            "nuclearity_vectors": nuclearity_vectors,
            "dist_bucket": dist_bucket,
            "ancestor_relation": ancestor_relation,
            "same_subtree": same_subtree,
            "parser_stats": torch.stack(parser_stats_rows, dim=0),
            "source_relation_bias_input": self.source_relation_embedding(source_relation_ids_tensor),
            "parser_eu": torch.stack(parser_eu_values).mean() if parser_eu_values else global_states.new_zeros(()),
        }

    def _build_structural_inputs_from_cache(
        self,
        graph: dict[str, Any],
        rst_cache: dict[str, Any],
        dtype: torch.dtype,
        device: torch.device,
        node_entropy_mode: str | None = None,
    ) -> dict[str, Any]:
        span_states = rst_cache["span_states"].to(device=device, dtype=dtype)
        edu_states = rst_cache["edu_states"].to(device=device, dtype=dtype)
        dist_bucket = rst_cache["tree_dist"].to(device=device, dtype=torch.long).clamp(max=self.tree_dist_cap)
        ancestor_relation = rst_cache["ancestor_relation"].to(device=device, dtype=torch.long)
        same_subtree = rst_cache["same_subtree"].to(device=device, dtype=torch.long)
        path_vectors = self._path_vectors_from_ids(
            rst_cache["branch_paths"],
            rst_cache["nuc_paths"],
            device=device,
            dtype=dtype,
        )
        node_inputs = []
        parser_vectors: dict[int, torch.Tensor] = {}
        nuclearity_vectors: dict[int, torch.Tensor] = {}
        parser_stats_rows = []
        source_relation_ids = []
        parser_eu_values = []
        stats_overrides = self._node_parser_stats_overrides(graph, node_entropy_mode=node_entropy_mode)

        for node in graph["nodes"]:
            if node["kind"] == "internal":
                parser_vec, stats = self._parser_observable(
                    node,
                    device=device,
                    dtype=dtype,
                    stats_override=stats_overrides.get(int(node["index"])),
                )
                nuc_id = torch.tensor(self.nuclearity_id(node.get("nuclearity_label")), device=device, dtype=torch.long)
                source_rel_id = self.relation_id(node.get("relation_label"))
                parser_vectors[int(node["parser_id"])] = parser_vec
                nuclearity_vectors[int(node["parser_id"])] = self._maybe_nuclearity_embedding(nuc_id)
                parser_eu_values.append(stats[4])
            else:
                parser_vec, stats = self._leaf_parser_observable(device=device, dtype=dtype)
                nuc_id = torch.tensor(NUCLEARITY_TO_ID["<LEAF>"], device=device, dtype=torch.long)
                source_rel_id = self.leaf_relation_id
            topology = torch.tensor(node["topology"], device=device, dtype=dtype)
            if not self.use_topology_features:
                topology = torch.zeros_like(topology)
            node_inputs.append(
                torch.cat(
                    [
                        self._maybe_span_state(span_states[int(node["index"])]),
                        parser_vec,
                        self._maybe_nuclearity_embedding(nuc_id),
                        topology,
                        self._maybe_path_vector(path_vectors[int(node["index"])]),
                    ],
                    dim=-1,
                )
            )
            parser_stats_rows.append(stats)
            source_relation_ids.append(source_rel_id)

        source_relation_ids_tensor = torch.tensor(source_relation_ids, device=device, dtype=torch.long)
        zero = span_states.new_zeros(())
        if not self.use_span_states:
            edu_states = torch.zeros_like(edu_states)
        return {
            "node_hidden": self.node_input_projection(torch.stack(node_inputs, dim=0)),
            "edu_states": edu_states,
            "parser_vectors": parser_vectors,
            "nuclearity_vectors": nuclearity_vectors,
            "dist_bucket": dist_bucket,
            "ancestor_relation": ancestor_relation,
            "same_subtree": same_subtree,
            "parser_stats": torch.stack(parser_stats_rows, dim=0),
            "source_relation_bias_input": self.source_relation_embedding(source_relation_ids_tensor),
            "parser_eu": torch.stack(parser_eu_values).mean() if parser_eu_values else zero,
        }

    def _forward_one(
        self,
        graph: dict[str, Any],
        global_states: torch.Tensor | None,
        doc_cls: torch.Tensor,
        edu_to_subwords: torch.Tensor,
        edu_mask: torch.Tensor,
        rst_cache: dict[str, Any] | None = None,
        node_entropy_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        needs_structure = self.needs_structure
        needs_node_structure = self.needs_node_structure
        h_text = self.text_projection(doc_cls)
        if not self.use_text_branch:
            h_text = torch.zeros_like(h_text)

        doc_parser_eu = h_text.new_zeros(())
        if needs_node_structure:
            if rst_cache is not None:
                structural = self._build_structural_inputs_from_cache(
                    graph=graph,
                    rst_cache=rst_cache,
                    dtype=doc_cls.dtype,
                    device=doc_cls.device,
                    node_entropy_mode=node_entropy_mode,
                )
            elif global_states is None:
                raise ValueError("Structural branches require global subword states, but window encoding was skipped.")
            else:
                structural = self._build_structural_inputs(
                    graph=graph,
                    global_states=global_states,
                    edu_to_subwords=edu_to_subwords,
                    edu_mask=edu_mask,
                    dtype=doc_cls.dtype,
                    node_entropy_mode=node_entropy_mode,
                )
            doc_parser_eu = structural["parser_eu"]
            if self.use_treeformer_branch:
                h_treeformer, attention_weights = self.treeformer(
                    hidden_states=structural["node_hidden"],
                    root_index=int(graph["root_index"]),
                    dist_bucket=structural["dist_bucket"],
                    ancestor_relation=structural["ancestor_relation"],
                    same_subtree=structural["same_subtree"],
                    source_relation_bias_input=structural["source_relation_bias_input"],
                    parser_stats=structural["parser_stats"],
                )
            else:
                h_treeformer = h_text.new_zeros(2 * h_text.size(0))
                attention_weights = h_text.new_zeros(0)

            if self.use_comp_branch:
                h_comp = self.tree_composer(
                    graph=graph,
                    edu_states=structural["edu_states"],
                    parser_vectors=structural["parser_vectors"],
                    nuclearity_vectors=structural["nuclearity_vectors"],
                )
            else:
                h_comp = h_text.new_zeros(h_text.size(0))
        else:
            h_treeformer = h_text.new_zeros(2 * h_text.size(0))
            h_comp = h_text.new_zeros(h_text.size(0))
            attention_weights = h_text.new_zeros(0)

        profile = (
            self._doc_profile(
                graph,
                dtype=doc_cls.dtype,
                device=doc_cls.device,
                node_entropy_mode=node_entropy_mode,
            )
            if self.use_parser_profile
            else doc_cls.new_zeros(DOC_PROFILE_DIM)
        )
        structural_repr = torch.cat([h_treeformer, h_comp, profile], dim=-1)
        doc_feature = doc_cls if self.use_text_branch else doc_cls.new_zeros(self.encoder_hidden_size)
        final = torch.cat([doc_feature, structural_repr], dim=-1) if needs_structure else doc_feature
        logits = self.ce_head(final)
        ce_probs = torch.softmax(logits, dim=-1)
        predictive_entropy = -(
            ce_probs.clamp_min(1.0e-12) * ce_probs.clamp_min(1.0e-12).log()
        ).sum() / math.log(float(self.num_labels))
        probs = ce_probs
        return {
            "logits": logits,
            "text_logits": logits,
            "rst_logits": logits.new_zeros(self.num_labels),
            "probs": probs,
            "ce_probs": ce_probs,
            "pred": probs.argmax(dim=-1),
            "uncertainty_mass": predictive_entropy,
            "doc_parser_eu": doc_parser_eu,
            "node_attention_entropy": -(
                attention_weights.clamp_min(1.0e-12) * attention_weights.clamp_min(1.0e-12).log()
            ).sum()
            if attention_weights.numel()
            else h_text.new_zeros(()),
        }

    def forward(self, batch, node_entropy_mode: str | None = None) -> dict[str, Any]:
        encode_windows = self.needs_node_structure
        if getattr(batch, "doc_repr", None) is not None and batch.doc_repr.numel() > 0:
            doc_cls = batch.doc_repr
            global_states = batch.global_states if encode_windows else None
        else:
            global_states, doc_cls = self.document_encoder(
                doc_input_ids=batch.doc_input_ids,
                doc_attention_mask=batch.doc_attention_mask,
                window_input_ids=batch.window_input_ids,
                window_attention_mask=batch.window_attention_mask,
                window_to_global=batch.window_to_global,
                encode_windows=encode_windows,
            )
        outputs = [
            self._forward_one(
                graph=graph,
                global_states=(
                    global_states[index][batch.global_state_mask[index]]
                    if global_states is not None and getattr(batch, "global_state_mask", None) is not None
                    else global_states[index]
                    if global_states is not None
                    else None
                ),
                doc_cls=doc_cls[index],
                edu_to_subwords=batch.edu_to_subwords[index],
                edu_mask=batch.edu_mask[index],
                rst_cache=batch.rst_tensors[index] if getattr(batch, "rst_tensors", None) else None,
                node_entropy_mode=node_entropy_mode,
            )
            for index, graph in enumerate(batch.graphs)
        ]
        return {
            "logits": torch.stack([item["logits"] for item in outputs], dim=0),
            "text_logits": torch.stack([item["text_logits"] for item in outputs], dim=0),
            "rst_logits": torch.stack([item["rst_logits"] for item in outputs], dim=0),
            "probs": torch.stack([item["probs"] for item in outputs], dim=0),
            "ce_probs": torch.stack([item["ce_probs"] for item in outputs], dim=0),
            "preds": torch.stack([item["pred"] for item in outputs], dim=0),
            "uncertainty_mass": torch.stack([item["uncertainty_mass"] for item in outputs], dim=0),
            "doc_parser_eu": torch.stack([item["doc_parser_eu"] for item in outputs], dim=0),
            "node_attention_entropy": torch.stack([item["node_attention_entropy"] for item in outputs], dim=0),
        }

    def forward_sota(
        self,
        doc_vectors: torch.Tensor,
        graphs: list[dict[str, Any]] | None = None,
        rst_tensors: list[dict[str, Any] | None] | None = None,
        node_entropy_mode: str | None = None,
    ) -> dict[str, Any]:
        """Forward path for the SOTA-compatible runner.

        `doc_vectors` are the frozen document representations produced by
        EntyRelCoh's own `SentDataset`/`fusion_vectors` cache.  When structural
        branches are disabled, this is exactly the SOTA text-only input to the
        same `BaseClassifer` head.  When structural branches are enabled, the
        only additional information is the fixed RST cache looked up by
        `text_id`.
        """
        needs_structure = self.needs_structure
        needs_node_structure = self.needs_node_structure
        if needs_structure:
            if graphs is None:
                raise ValueError("SOTA-compatible structured mode requires graphs.")
            if needs_node_structure and rst_tensors is None:
                raise ValueError("SOTA-compatible TreeFormer mode requires cached RST tensors.")
            outputs = [
                self._forward_one(
                    graph=graph,
                    global_states=None,
                    doc_cls=doc_vectors[index],
                    edu_to_subwords=doc_vectors.new_zeros(0, dtype=torch.long),
                    edu_mask=torch.zeros(0, dtype=torch.bool, device=doc_vectors.device),
                    rst_cache=rst_tensors[index] if rst_tensors is not None else None,
                    node_entropy_mode=node_entropy_mode,
                )
                for index, graph in enumerate(graphs)
            ]
            return {
                "logits": torch.stack([item["logits"] for item in outputs], dim=0),
                "text_logits": torch.stack([item["text_logits"] for item in outputs], dim=0),
                "rst_logits": torch.stack([item["rst_logits"] for item in outputs], dim=0),
                "probs": torch.stack([item["probs"] for item in outputs], dim=0),
                "ce_probs": torch.stack([item["ce_probs"] for item in outputs], dim=0),
                "preds": torch.stack([item["pred"] for item in outputs], dim=0),
                "uncertainty_mass": torch.stack([item["uncertainty_mass"] for item in outputs], dim=0),
                "doc_parser_eu": torch.stack([item["doc_parser_eu"] for item in outputs], dim=0),
                "node_attention_entropy": torch.stack([item["node_attention_entropy"] for item in outputs], dim=0),
            }

        batch_size = int(doc_vectors.size(0))
        doc_feature = doc_vectors if self.use_text_branch else doc_vectors.new_zeros(doc_vectors.shape)
        logits = self.ce_head(doc_feature)
        probs = torch.softmax(logits, dim=-1)
        predictive_entropy = -(
            probs.clamp_min(1.0e-12) * probs.clamp_min(1.0e-12).log()
        ).sum(dim=-1) / math.log(float(self.num_labels))
        return {
            "logits": logits,
            "text_logits": logits,
            "rst_logits": logits.new_zeros(batch_size, self.num_labels),
            "probs": probs,
            "ce_probs": probs,
            "preds": probs.argmax(dim=-1),
            "uncertainty_mass": predictive_entropy,
            "doc_parser_eu": logits.new_zeros(batch_size),
            "node_attention_entropy": logits.new_zeros(batch_size),
        }
