"""Checkpoint-backed native prediction adapters; no fitting or metric evaluation."""

import importlib
from pathlib import Path
import random
import sys

import numpy as np


def _torch(device, seed):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch


def _controls(data, minimum=1):
    if "controls" not in data or "feature_names" not in data:
        raise ValueError("Prepared native inputs require real controls and ordered feature_names")
    x = np.asarray(data["controls"])
    genes = data["feature_names"].astype(str)
    if x.ndim != 2 or x.shape[1] != len(genes) or len(x) < minimum or not np.isfinite(x).all():
        raise ValueError("Real control cells and ordered gene names are required")
    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate gene identifiers")
    baseline = np.asarray(data.get("control_mean", x.mean(axis=0)))
    if baseline.shape != (len(genes),) or not np.isfinite(baseline).all():
        raise ValueError("Invalid observed control mean")
    return x, genes, baseline


def morph(checkpoint, data, args):
    torch = _torch(args.device, args.seed)
    if args.architecture == "benchmark_mean_conditioned":
        from .morph.benchmark_model import MORPHModel
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "mu.weight" not in state:
            raise ValueError("Checkpoint does not match benchmark_mean_conditioned; do not interchange MORPH variants")
        g, d = state["ce.0.weight"].shape[1], state["pe.0.weight"].shape[1]
        model = MORPHModel(g, d, state["mu.weight"].shape[0])
        model.load_state_dict(state, strict=True)
        model.to(args.device).eval()
        features = data["feature_names"].astype(str)
        embeddings = data.get("descriptors", data.get("features"))
        baseline = np.asarray(data["conditioning_mean"], dtype=np.float32)
        if (len(features) != g or baseline.shape != (g,) or embeddings is None or
                embeddings.shape != (len(data["test_perts"]), d)):
            raise ValueError("Mean-conditioned MORPH needs matching descriptors and the fitted conditioning_mean")
        predictions = []
        with torch.inference_mode():
            x = torch.as_tensor(baseline[None], device=args.device)
            for vector in embeddings:
                prediction = model(x, torch.as_tensor(vector[None], dtype=torch.float32, device=args.device))
                predictions.append(prediction.cpu().numpy()[0] - baseline)
        return np.stack(predictions), features, {"architecture": args.architecture,
            "control_subtraction": "recorded_fitted_conditioning_mean"}
    if args.architecture != "upstream_pickled_model":
        raise ValueError("Select the exact MORPH architecture from the artifact record")
    if not args.trust_checkpoint or not args.source_dir:
        raise ValueError("Upstream pickled MORPH requires --trust_checkpoint and --source_dir")
    source = Path(args.source_dir).resolve()
    module_root = source / "morph" if (source / "morph/model.py").is_file() else source
    if not (module_root / "model.py").is_file():
        raise FileNotFoundError("MORPH source must contain model.py")
    sys.path.insert(0, str(module_root))
    module = importlib.import_module("model")
    if not Path(module.__file__).resolve().is_relative_to(module_root):
        raise ValueError("A different model module is already loaded")
    model = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.double().to(args.device).eval()
    x, features, baseline = _controls(data)
    descriptors = data["descriptors"]
    if descriptors.shape != (len(data["test_perts"]), int(model.c_dim)):
        raise ValueError("Descriptor axis differs from the trained MORPH model")
    rng = np.random.RandomState(args.seed)
    predictions = []
    with torch.inference_mode():
        for vector in descriptors:
            indices = rng.choice(len(x), min(args.control_samples, len(x)), replace=True)
            control = torch.as_tensor(x[indices], dtype=torch.float64, device=args.device)
            condition = torch.as_tensor(vector, dtype=torch.float64, device=args.device)[None].expand(len(indices), -1)
            predicted, *_ = model(control, condition, torch.full_like(condition, float("nan")))
            predictions.append(predicted.cpu().numpy().mean(0)-baseline)
    return np.stack(predictions), features, {"architecture": args.architecture,
        "control_subtraction": "observed_target_control_mean", "control_samples": args.control_samples}


def state(checkpoint, data, args):
    if not args.trust_checkpoint:
        raise ValueError("Lightning checkpoint loading requires --trust_checkpoint")
    torch = _torch(args.device, args.seed)
    from models.state.compat import install
    compatibility = install()
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    x, features, baseline = _controls(data, args.cell_set_len)
    labels = data["perturbation_labels"].astype(str)
    if labels.ndim != 1 or len(set(labels)) != len(labels):
        raise ValueError("The complete recorded one-hot label order is required")
    lookup = {name: i for i, name in enumerate(labels)}
    if not args.perturbation_map:
        raise ValueError("STATE requires the checkpoint's --perturbation_map; test labels alone do not define its encoding")
    mapping = torch.load(args.perturbation_map, map_location="cpu", weights_only=False)
    if not isinstance(mapping, dict) or {str(k) for k in mapping} != set(labels):
        raise ValueError("The recorded STATE perturbation map differs from prepared labels")
    for name, vector in mapping.items():
        value = vector.detach().cpu().numpy() if hasattr(vector, "detach") else np.asarray(vector)
        expected = np.zeros(len(labels), dtype=value.dtype)
        expected[lookup[str(name)]] = 1
        if value.shape != expected.shape or not np.array_equal(value, expected):
            raise ValueError("STATE one-hot ordering differs from the saved map")
    if set(data["test_perts"].astype(str)) - set(lookup):
        raise ValueError("STATE cannot predict labels absent from its trained one-hot map")
    model = StateTransitionPerturbationModel.load_from_checkpoint(
        str(checkpoint), map_location=args.device, weights_only=False)
    model.eval().to(args.device)
    predictions = []
    with torch.inference_mode():
        for perturbation in data["test_perts"].astype(str):
            batches = []
            for bi in range(args.control_batches):
                indices = np.random.RandomState(args.seed*1000+bi).choice(len(x), args.cell_set_len, replace=False)
                control = torch.as_tensor(x[indices].astype(np.float32), device=args.device)
                onehot = torch.zeros(args.cell_set_len, len(labels), device=args.device)
                onehot[:, lookup[perturbation]] = 1.
                batch = {"ctrl_cell_emb": control, "pert_emb": onehot,
                         "batch": torch.zeros(args.cell_set_len, dtype=torch.long, device=args.device)}
                output = model.forward(batch, padded=False)
                batches.append(output.cpu().numpy().mean(0))
            predictions.append(np.mean(batches, axis=0)-baseline)
    return np.stack(predictions), features, {"architecture": "state_transition",
        "control_subtraction": "observed_target_control_mean",
        "cell_set_len": args.cell_set_len, "control_batches": args.control_batches,
        "recorded_helper_sha256": compatibility["recorded_utils_sha256"]}


def scgpt(checkpoint, data, args):
    if not args.vocab or not args.model_config:
        raise ValueError("scGPT requires the exact --vocab and --model_config used by the checkpoint")
    torch = _torch(args.device, args.seed)
    import anndata as ad
    import pandas as pd
    from scipy import sparse
    from gears.utils import create_cell_graph_dataset_for_prediction
    from torch_geometric.loader import DataLoader
    from scgpt.model import TransformerGenerator
    from scgpt.model.model import ExprDecoder
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from reproduction.io import read_json, sha256
    x, features, baseline = _controls(data)
    vocab = GeneVocab.from_file(args.vocab)
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    unknown = [name for name in features if name not in vocab]
    policy_path = getattr(args, "token_policy", None)
    policy = None
    if policy_path:
        from models.token_policy import validate_policy
        policy = validate_policy(policy_path, args.vocab, features.tolist())
        vocab.set_default_index(policy["default_index"])
    elif unknown:
        raise ValueError("Genes are absent from the vocabulary. Historical scGPT inputs require "
                         "their explicit --token_policy companion; no genes are silently dropped.")
    gene_ids = np.asarray(vocab(features.tolist()), dtype=int)
    config = read_json(args.model_config)
    width = config.get("embsize", 512)
    model = TransformerGenerator(
        ntoken=len(vocab), d_model=width, nhead=config.get("nheads", 8),
        d_hid=config.get("d_hid", 512), nlayers=config.get("nlayers", 12),
        nlayers_cls=config.get("n_layers_cls", 3), n_cls=1, vocab=vocab,
        dropout=config.get("dropout", .2), pad_token="<pad>", pad_value=0,
        pert_pad_id=2, do_mvc=False, cell_emb_style="cls",
        mvc_decoder_style="inner product, detach")

    class ExprDecoderCompat(ExprDecoder):
        def forward(self, inputs, values=None):
            return super().forward(inputs)

    model.decoder = ExprDecoderCompat(d_model=width, explicit_zero_prob=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(args.device).eval()
    controls = ad.AnnData(X=sparse.csr_matrix(x),
        obs=pd.DataFrame({"condition": ["ctrl"]*len(x)}, index=[f"ctrl{i}" for i in range(len(x))]),
        var=pd.DataFrame({"gene_name": features}, index=features))
    predictions = []
    amp = args.device.startswith("cuda")
    with torch.inference_mode():
        for condition in data["test_perts"].astype(str):
            genes = [name for name in condition.split("+") if name != "ctrl"]
            if not genes or set(genes)-set(features):
                raise ValueError(f"Perturbation is not represented on the input gene axis: {condition}")
            graphs = create_cell_graph_dataset_for_prediction(
                genes, controls, features.tolist(), args.device, num_samples=len(controls))
            loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
            values = [model.pred_perturb(batch, "all", gene_ids=gene_ids, amp=amp) for batch in loader]
            predictions.append(torch.cat(values).cpu().numpy().mean(0)-baseline)
    return np.stack(predictions), features, {"architecture": "fine_tuned_transformer_expr_decoder",
        "control_subtraction": "observed_control_mean", "amp": amp,
        "gene_policy": "recorded_vocab_default" if policy else "strict_vocab_and_input_axis",
        "unknown_gene_count": len(unknown),
        "token_policy_sha256": sha256(policy_path) if policy_path else None}


def biolord(checkpoint, data, args):
    if not args.trust_checkpoint or not args.training_data or not args.go_features or not args.target:
        raise ValueError("BioLORD requires --trust_checkpoint, --training_data, --go_features, and --target")
    torch = _torch(args.device, args.seed)
    import anndata as ad
    import biolord as native
    from scvi import REGISTRY_KEYS
    training = ad.read_h5ad(args.training_data)
    if "cell_line" not in training.obs or "role" not in training.obs:
        raise ValueError("Use the frozen prepared BioLORD AnnData with membership roles")
    controls = np.flatnonzero((training.obs["cell_line"].astype(str).to_numpy() == args.target) &
                              (training.obs["role"].astype(str).to_numpy() == "target_control"))
    target_rows = training.obs["cell_line"].astype(str).to_numpy() == args.target
    if len(controls) != 1 or target_rows.sum() != 1:
        raise ValueError("Prepared target context must contain exactly one control condition mean")
    model_dir = Path(checkpoint).parent if Path(checkpoint).is_file() else Path(checkpoint)
    if args.device != "cpu":
        raise ValueError("The BioLORD checkpoint replay adapter uses --device cpu")
    model = native.Biolord.load(str(model_dir), adata=training, accelerator="cpu", device=1)
    if model.module.latent_codes.embed:
        raise ValueError("This adapter only supports the GO/cell-line unknown_attributes=False protocol")
    if (set(model.categorical_attributes_map) != {"cell_line"} or
            set(model.ordered_attributes_map) != {"perturbation_neighbors1"}):
        raise ValueError("The checkpoint does not match GO/cell-line conditioning")
    if not set(controls).issubset(set(model.train_indices)):
        raise ValueError("Target controls were not in the fitted training split")
    dataset = model.get_dataset(model.adata, indices=controls)
    actual = dataset[REGISTRY_KEYS.INDICES_KEY].cpu().numpy().ravel()
    if not np.array_equal(actual, controls):
        raise ValueError("Registered target-control indices changed")
    dataset = {key: value.to(model.device) for key, value in dataset.items()}
    with np.load(args.go_features, allow_pickle=False) as saved:
        descriptors = dict(zip(saved["perturbations"].astype(str), saved["descriptors"]))
    key = "perturbation_neighbors1"
    names = data["test_perts"].astype(str)
    if set(names)-set(descriptors) or "non-targeting" not in descriptors:
        raise ValueError("GO descriptors are missing; no perturbations are silently dropped")

    def expression(descriptor):
        tensors = dict(dataset)
        attribute = torch.as_tensor(descriptor, dtype=tensors[key].dtype, device=model.device)
        tensors[key] = attribute[None].expand(len(controls), -1)
        value, _ = model.module.get_expression(tensors)
        if value.ndim == 1 and len(controls) == 1:
            value = value.unsqueeze(0)
        if tuple(value.shape) != (len(controls), model.adata.n_vars):
            raise ValueError("BioLORD expression lost its gene axis")
        return value.detach().cpu().numpy().mean(0)

    model.module.eval()
    with torch.inference_mode():
        baseline = expression(descriptors["non-targeting"])
        predictions = np.stack([expression(descriptors[p])-baseline for p in names]).astype(np.float32)
    return predictions, model.adata.var_names.to_numpy().astype(str), {
        "architecture": "go_cellline_condition_means",
        "control_subtraction": "reconstructed_target_control_mean"}


def scgen(checkpoint, data, args):
    if not args.trust_checkpoint or not args.target:
        raise ValueError("Native binary scGen loading requires --trust_checkpoint and --target")
    from reproduction.io import read_json, sha256
    directory = Path(checkpoint)
    protocol = read_json(directory / "protocol.json")
    if (protocol["protocol"] != "scgen_binary_target_controls_real_cell_labels_native_all_v1"
            or protocol["target"] != args.target
            or data["test_perts"].astype(str).tolist() != [protocol["perturbation"]]):
        raise ValueError("A binary scGen checkpoint is valid only for its recorded target/perturbation")
    if sha256(directory / "model.pt") != protocol["model_sha256"]:
        raise ValueError("Native scGen checkpoint changed")
    _torch(args.device, args.seed)
    import scgen as native
    model = native.SCGEN.load(str(directory), use_gpu=args.device if args.device.startswith("cuda") else False)
    model.module.eval()
    target_rows = model.adata.obs["cell_state"].astype(str).to_numpy() == args.target
    if not target_rows.any() or not np.all(model.adata.obs.loc[target_rows, "role"] == "target_control"):
        raise ValueError("The saved binary AnnData contains forbidden target perturbed cells")
    predicted, _ = model.predict(ctrl_key="control", stim_key="stimulated", celltype_to_predict=args.target)
    controls = model.adata[model.adata.obs["role"] == "target_control"]
    values = np.asarray(predicted.X.mean(0)-controls.X.mean(0), dtype=np.float32).reshape(1, -1)
    return values, model.adata.var_names.to_numpy().astype(str), {
        "architecture": "binary_per_perturbation", "control_subtraction": "observed_target_control_mean",
        "target": args.target, "perturbation": protocol["perturbation"]}
