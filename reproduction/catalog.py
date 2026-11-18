"""Explicit figure selection; archived variants are not silently interchanged."""

CELLS = ("k562", "rpe1", "hepg2", "jurkat")
CONTEXT = "benchmarks/context_generalization"
BIOLORD = f"{CONTEXT}/results/biolord/three_to_one_cellline_go_author_v1_exportfixed_v2"
SCGEN = f"{CONTEXT}/results/scgen/three_to_one_author_aligned_v1_numpycompat"
CALIBRATION = f"{CONTEXT}/results/figure4c_author_aligned_v2_retry1"
MATCHING = f"{CONTEXT}/results/component_matching_assembled_v1"

# source JSON, renderer, optional metric fields.
RENDERABLE = {
    "fig1_d": ("final/hallmark/results.json", "heatmap", ()),
    "fig1_dsup": ("paper/figure1/fig1dsup_identity_variance_results.json", "fields", ("pct",)),
    "fig1_e": ("paper/figure1/fig1e_ctrl_space_results.json", "control_space", ()),
    "fig1_f": ("paper/figure1/fig1f_anova_results.json", "stacked", ()),
    "fig2_a": ("paper/figure2/fig2a_within_cl_pref_results.json", "models", ()),
    "fig2_b": ("paper/figure2/fig2b_depmap_pairwise_results.json", "fields", ("rho_full", "rho_resid")),
    "fig2_c": ("paper/figure2/fig2c_ridge_projection_results.json", "fields", ("rho_raw", "rho_proj")),
    "fig2_d": ("paper/figure2/fig2d_rank_ablation_results.json", "ranks", ()),
    "fig2_d2": ("paper/figure2/fig2d_rank_ablation_results.json", "ranks", ()),
    "fig2_e": ("paper/figure2/fig2e_morph_mechanism_results.json", "mechanism", ()),
    "fig3_crosscl_transfer": ("final/figure3/results.json", "transfer", ()),
    "fig3_c": ("paper/figure3/fig3c_double_unseen_results.json", "models", ()),
    "fig3_d": ("paper/figure3/fig3d_target_landscape_results.json", "landscape", ()),
    "fig3_e": ("paper/figure3/fig3e_basis_vs_transfer_results.json", "basis", ()),
    "fig3_enrichment_barplot": ("benchmark/enrichment_plot_data.json", "enrichment", ()),
    "fig4_a": ("paper/figure4/fig4a_within_decomp_results.json", "attribution", ("alpha", "eps")),
    "fig4_b": ("final/components/results.json", "components_source", ()),
    "fig4_c": ("final/components/results.json", "gamma_matching", ()),
    "fig4_calibration": ("final/figure4c/results.json", "calibration", ()),
    "fig4_crosscl_decomp": ("paper/figure4/fig4_crosscl_decomp_results.json", "attribution", ("alpha", "eps")),
    "fig4_d": ("final/components/results.json", "gamma_matching", ()),
}

FINAL_REFERENCE_PREFIXES = {
    "fig4_a": "final/figure4a/",
    "fig1_d": "final/hallmark/",
    "fig4_d": "final/figure4d/",
    "fig3_a": "final/figure3/",
    "fig3_crosscl_transfer": "final/figure3/",
    "fig4_b": "final/figure4b/",
    "fig4_c": "final/figure4d/",
    "fig4_calibration": "final/figure4c_clean/",
}

# Exact historical vector assets. Copying these is NOT analysis recomputation.
REFERENCE_PREFIXES = {
    "fig1_b": ("figure1/fig1b_template_variance",),
    "fig1_c": ("figure1/fig1c_erank_slope", "figure1/fig1_erank_slope"),
    "fig1_dsup": ("figure1/fig1dsup_identity_variance",),
    "fig1_e": ("figure1/fig1e_ctrl_space",),
    "fig1_f": ("figure1/fig1f_anova_stacked",),
    "fig1_suppl_decomp": ("decomp_supplementary/decomp_supp_",),
    "fig1_supplementary_full": ("figure_1_supplementary/s1",),
    "fig2_a": ("figure2/fig2a_",),
    "fig2_b": ("figure2/fig2b_",),
    "fig2_c": ("figure2/fig2c_",),
    "fig2_d": ("figure2/fig2d_",),
    "fig2_d2": ("figure2/fig2d_",),
    "fig2_e": ("figure2/fig2e_",),
    "fig2_supp2_1": ("supplemetary_figure_2_2/s2",),
    "fig2_supp2_2": ("supplementary_figure_2_3/s2",),
    "fig3_b": ("figure3/fig3b_",),
    "fig3_c": ("figure3/fig3c_",),
    "fig3_d": ("figure3/fig3d_",),
    "fig3_e": ("figure3/fig3e_basis_vs_transfer",),
    "fig3_source_vs_depmap": ("figure3/fig3_source_vs_depmap",),
    "fig3_morph_double_unseen_combo": ("figure3/fig3_morph_double_unseen_clean",),
    "fig3_supp1": ("figure3_supplementary/s3",),
    "fig4_crosscl_decomp": ("figure4/fig4_crosscl_decomp",),
    "fig4_supplementary": ("figure4_supplementary/s4",),
    "fig_pathway_crosscl": ("figure_pathway/crosscl_pathway_",),
    "fig_tahoe_panels": ("figure_tahoe/fig_tahoe_",),
    "fig_tahoe_b_saturation": ("figure_tahoe/fig_tahoe_b_",),
    "fig_tahoe_c_controls": ("figure_tahoe/fig_tahoe_c_",),
    "fig3_supp1_heatmaps": ("figure3_supplementary/s3a_",),
    "fig3_supp1_basis": ("figure3_supplementary/s3d_",),
    "fig3_supp1_baselines": ("figure3_supplementary/s3e_",),
    "fig1_supp_alignment": ("figure_1_supplementary/s1b_",),
    "fig1_supp_anova": ("figure_1_supplementary/s1f_",),
    "fig1_supp_hallmark": ("figure_1_supplementary/s1d_",),
    "fig1_supp_noise": ("figure_1_supplementary/s1e_",),
    "fig2_supp_deciles": ("supplemetary_figure_2_2/s2a_",),
    "fig2_supp_scatter": ("supplemetary_figure_2_2/s2c_",),
    "fig3_supp1_slopes": ("figure3_supplementary/s3b_",),
    "fig3_supp1_calibration": ("figure3_supplementary/s3c_",),
}

LIMITATIONS = {}


def figure_ids():
    return sorted(set(RENDERABLE) | set(REFERENCE_PREFIXES)
                  | set(FINAL_REFERENCE_PREFIXES) | set(LIMITATIONS))
