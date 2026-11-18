MAX_TERMS_PER_SIDE = 7

def remove_redundant(terms, term_to_genes, max_overlap=0.5):
    """Remove redundant GO terms by gene overlap. Keep better FDR."""
    if not terms:
        return terms

    # Sort by FDR (best first)
    sorted_terms = sorted(terms, key=lambda x: x["fdr"])
    kept = []
    kept_gene_sets = []

    for t in sorted_terms:
        t_genes = term_to_genes.get(t["term"], set())
        if not t_genes:
            continue

        redundant = False
        for kg in kept_gene_sets:
            overlap = len(t_genes & kg) / min(len(t_genes), len(kg))
            if overlap > max_overlap:
                redundant = True
                break

        if not redundant:
            kept.append(t)
            kept_gene_sets.append(t_genes)

        if len(kept) >= MAX_TERMS_PER_SIDE:
            break

    return kept
