"""Tiny synthetic API checks for the separately installed native SDKs."""

import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


class LockedNativeRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("scgpt"), "scGPT runtime required")
    def test_scgpt_preserves_vocabulary_ids_and_unknown_default(self):
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        mapping = {"G2": 1, "<eoc>": 4, "G1": 0, "<pad>": 2, "<cls>": 3}
        vocab = GeneVocab.from_dict(mapping)
        self.assertEqual(vocab.get_stoi(), mapping)
        self.assertEqual(vocab.get_itos(), ["G1", "G2", "<pad>", "<cls>", "<eoc>"])
        self.assertEqual(vocab(["G2", "UNKNOWN", "G1"]), [1, 2, 0])
        self.assertEqual(len(vocab), 5)
        vocab.set_default_index(1)
        self.assertEqual(vocab(["UNKNOWN"]), [1])

    @unittest.skipUnless(importlib.util.find_spec("state"), "STATE runtime required")
    def test_recorded_state_helper_constructs_and_runs_a_tiny_forward(self):
        import torch
        from transformers import GPT2Config
        from models.state.compat import install
        install()
        from state.tx.models.utils import GPT2BidirectionalModel
        torch.set_num_threads(1)
        model = GPT2BidirectionalModel(GPT2Config(
            vocab_size=8, n_embd=8, n_head=2, n_layer=1, n_positions=4)).eval()
        with torch.inference_mode():
            output = model(inputs_embeds=torch.ones(1, 4, 8))
        self.assertEqual(tuple(output.last_hidden_state.shape), (1, 4, 8))
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())
        from lightning import __version__ as lightning_version
        from state.tx.models.state_transition import StateTransitionPerturbationModel
        from models.native_inference import state
        model = StateTransitionPerturbationModel(
            input_dim=4, hidden_dim=8, output_dim=4, pert_dim=2, gene_dim=4, hvg_dim=4,
            gene_decoder_bool=False, embed_key="X_hvg", cell_set_len=4,
            n_encoder_layers=1, n_decoder_layers=1,
            transformer_backbone_kwargs={"hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 4,
                "use_cache": False}).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, mapping = root / "model.ckpt", root / "map.pt"
            torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                        "pytorch-lightning_version": lightning_version}, checkpoint)
            torch.save({"P0": torch.tensor([1., 0.]), "P1": torch.tensor([0., 1.])}, mapping)
            data = {"controls": np.ones((8, 4), dtype=np.float32),
                    "feature_names": np.asarray(["g0", "g1", "g2", "g3"]),
                    "test_perts": np.asarray(["P1"]), "perturbation_labels": np.asarray(["P0", "P1"])}
            args = argparse.Namespace(trust_checkpoint=True, device="cpu", seed=0,
                                      perturbation_map=mapping, cell_set_len=4, control_batches=1)
            values, names, _ = state(checkpoint, data, args)
            self.assertEqual(values.shape, (1, 4))
            self.assertEqual(names.tolist(), ["g0", "g1", "g2", "g3"])
            self.assertTrue(np.isfinite(values).all())

    @unittest.skipUnless(importlib.util.find_spec("scgpt"), "scGPT runtime required")
    def test_scgpt_gears_tiny_checkpoint_and_explicit_token_policy(self):
        import torch
        from scgpt.model import TransformerGenerator
        from scgpt.model.model import ExprDecoder
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        from models.native_inference import scgpt
        from models.token_policy import recorded_policy
        from reproduction.io import write_json
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = {"G1": 0, "G2": 1, "<pad>": 2, "<cls>": 3, "<eoc>": 4}
            vocab_path = root / "vocab.json"
            write_json(vocab_path, mapping)
            vocab = GeneVocab.from_file(vocab_path)
            config = {"embsize": 8, "nheads": 2, "d_hid": 16, "nlayers": 1,
                      "n_layers_cls": 1, "dropout": 0.0}
            config_path = root / "config.json"
            write_json(config_path, config)
            model = TransformerGenerator(ntoken=len(vocab), d_model=8, nhead=2, d_hid=16,
                nlayers=1, nlayers_cls=1, n_cls=1, vocab=vocab, dropout=0,
                pad_token="<pad>", pad_value=0, pert_pad_id=2, do_mvc=False,
                cell_emb_style="cls", mvc_decoder_style="inner product, detach")
            model.decoder = ExprDecoder(d_model=8, explicit_zero_prob=False)
            checkpoint = root / "weights.pt"
            torch.save(model.state_dict(), checkpoint)
            genes = np.asarray(["G1", "G2", "UNKNOWN"])
            policy_path = root / "token_policy.json"
            write_json(policy_path, recorded_policy(vocab_path, genes.tolist()))
            data = {"test_perts": np.asarray(["G1"]), "feature_names": genes,
                    "controls": np.ones((2, 3), dtype=np.float32)}
            args = argparse.Namespace(vocab=vocab_path, model_config=config_path,
                token_policy=policy_path, device="cpu", seed=0, batch_size=1)
            values, names, metadata = scgpt(checkpoint, data, args)
            self.assertEqual(values.shape, (1, 3))
            np.testing.assert_array_equal(names, genes)
            self.assertEqual(metadata["unknown_gene_count"], 1)
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
