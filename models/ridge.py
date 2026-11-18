"""Ridge and MLP baseline models for perturbation prediction."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


def train_ridge(V_train, D_train, alpha=10, n_pca=50, pca_seed=None):
    """Train Ridge regression with PCA preprocessing.

    Args:
        V_train: (n_train, d_features) input features (DepMap embeddings)
        D_train: (n_train, n_genes) target deltas
        alpha: Ridge regularization
        n_pca: number of PCA components (capped at n_train-1)

    Returns:
        predict_fn: function (V_test) -> predicted deltas
        train_centroid: mean delta of training perturbations
    """
    n_comp = min(n_pca, V_train.shape[0] - 1, V_train.shape[1])
    pca = PCA(n_comp, random_state=pca_seed).fit(V_train)
    ridge = Ridge(alpha=alpha).fit(pca.transform(V_train), D_train)
    train_centroid = D_train.mean(0)

    def predict_fn(V_test):
        return ridge.predict(pca.transform(V_test))

    predict_fn.checkpoint_arrays = {
        "pca_components": pca.components_.copy(), "pca_mean": pca.mean_.copy(),
        "ridge_coef": ridge.coef_.copy(), "ridge_intercept": np.asarray(ridge.intercept_).copy(),
    }
    predict_fn.checkpoint_metadata = {"kind": "ridge", "alpha": alpha, "n_pca": n_comp,
                                     "pca_seed": pca_seed, "pca_solver": pca._fit_svd_solver}
    return predict_fn, train_centroid


try:
    import torch
    import torch.nn as nn

    class MLPModel(nn.Module):
        """Simple MLP for perturbation prediction."""
        def __init__(self, d_in, n_genes, hidden_dims=(256, 256), dropout=0.0):
            super().__init__()
            layers = []
            prev = d_in
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU()])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, n_genes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def train_mlp(V_train, D_train, seed=0, n_pca=50, hidden_dims=(256, 256),
                  lr=5e-4, weight_decay=1e-5, epochs=400, batch_size=64,
                  device="cuda:0", pca_seed=None):
        """Train MLP with PCA preprocessing.

        Args:
            V_train: (n_train, d_features) input features
            D_train: (n_train, n_genes) target deltas
            seed: random seed
            n_pca: PCA components
            hidden_dims: MLP hidden layer sizes
            lr: learning rate
            weight_decay: L2 regularization
            epochs: training epochs
            batch_size: mini-batch size
            device: torch device

        Returns:
            predict_fn: function (V_test) -> predicted deltas (numpy)
            train_centroid: mean delta of training perturbations
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        n_comp = min(n_pca, V_train.shape[0] - 1, V_train.shape[1])
        pca = PCA(n_comp, random_state=pca_seed).fit(V_train)
        V_pca = pca.transform(V_train).astype(np.float32)

        model = MLPModel(n_comp, D_train.shape[1], hidden_dims).to(device).float()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        Vt = torch.tensor(V_pca, device=device)
        Dt = torch.tensor(D_train.astype(np.float32), device=device)
        n = len(V_train)

        model.train()
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                loss = nn.MSELoss()(model(Vt[idx]), Dt[idx])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        train_centroid = D_train.mean(0)

        def predict_fn(V_test):
            V_test_pca = pca.transform(V_test).astype(np.float32)
            with torch.no_grad():
                pred = model(torch.tensor(V_test_pca, device=device))
            return pred.cpu().numpy()

        arrays = {"pca_components": pca.components_.copy(), "pca_mean": pca.mean_.copy()}
        linear = [layer for layer in model.net if isinstance(layer, nn.Linear)]
        for index, layer in enumerate(linear):
            arrays[f"layer{index}_weight"] = layer.weight.detach().cpu().numpy().copy()
            arrays[f"layer{index}_bias"] = layer.bias.detach().cpu().numpy().copy()
        predict_fn.checkpoint_arrays = arrays
        predict_fn.checkpoint_metadata = {
            "kind": "mlp", "n_layers": len(linear), "n_pca": n_comp,
            "seed": seed, "epochs": epochs, "batch_size": batch_size,
            "hidden_dims": list(hidden_dims), "lr": lr, "weight_decay": weight_decay}
        predict_fn.checkpoint_metadata.update(pca_seed=pca_seed, pca_solver=pca._fit_svd_solver)
        return predict_fn, train_centroid

    HAS_TORCH = True

except ImportError:
    HAS_TORCH = False

    def train_mlp(*args, **kwargs):
        raise ImportError("PyTorch required for MLP training")
