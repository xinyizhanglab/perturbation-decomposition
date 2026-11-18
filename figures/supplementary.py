"""Restore plotting variables from saved supplementary result tables."""

import copy

import numpy as np

from reproduction.catalog import CELLS


SUPPORTED = (
    "fig1_supp_alignment", "fig1_supp_anova",
    "fig3_supp1_heatmaps", "fig3_supp1_basis", "fig3_supp1_baselines",
    "fig4_supplementary",
)
LABELS = ("K562", "RPE1", "HepG2", "Jurkat")


def prepare(identifier, data):
    if identifier not in SUPPORTED:
        raise ValueError(f"Unknown supplementary plotting adapter: {identifier}")
    if identifier == "fig1_supp_alignment":
        return {key: [data["s1b"][key][cell] for cell in CELLS]
                for key in ("cos_delta_pc1", "cos_ctrl_pc1")}
    if identifier == "fig1_supp_anova":
        group = data["s1f"]
        return {
            **{name: group["cross_dot_products"][key] for name, key in (
                ("cross_ab", "alpha_beta"), ("cross_ag", "alpha_gamma"), ("cross_bg", "beta_gamma"))},
            **{key: [group[key][cell] for cell in CELLS] for key in ("beta_per_cl", "gamma_per_cl")},
        }
    if identifier.startswith("fig3_supp1_"):
        if identifier == "fig3_supp1_baselines":
            return {"baseline_results": data["s3e"]}
        result = {
            "n_cl": len(CELLS), "cl_labels": list(LABELS),
            "ridge_matrix": np.asarray(data["s3a"]["ridge_matrix"], dtype=float),
            "mlp_matrix": np.asarray(data["s3a"]["mlp_matrix"], dtype=float),
        }
        if identifier == "fig3_supp1_basis":
            result.update(
                BASIS_RANK=20,
                overlap_matrix=np.asarray(data["s3d"]["overlap_matrix"], dtype=float),
                rho=data["s3d"]["spearman_rho"], p_val=data["s3d"]["spearman_p"],
            )
        return result

    within = copy.deepcopy(data["s4b_within_decomp"])
    for model in within.values():
        for entry in model.values():
            for key in ("t_frac", "e_frac"):
                entry[key] = np.float32(entry[key])
    result = {
        "models": ["Ridge", "MLP"], "within_decomp": within,
        "cross_decomp": data["s4c_cross_decomp"],
        "state_data": data["s4d_state"], "morph_data": data["s4d_morph"],
        "state_metrics": ["pref", "eps_corr", "beta_corr", "gamma_corr"],
        "state_settings": ["zeroshot", "calibrated"],
        "morph_settings_list": ["Source only", "Target ctrls", "+30% target"],
        "calib_results": data["s4e_calibration"],
        "calib_settings": ["Zero-shot", "Source +30%", "Target-only 30%"],
    }
    for key, variable in (("s4a_within_features", "within"), ("s4a_cross_features", "cross")):
        group = data[key]
        result[variable.upper() + "_FEATS"] = group["features"]
        result[variable + "_results"] = np.asarray([
            [group["values"][cell][feature] for feature in group["features"]] for cell in LABELS
        ], dtype=float)
    return result
