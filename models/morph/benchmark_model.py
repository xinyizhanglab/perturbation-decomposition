"""Mean-conditioned MORPH architecture used by the within-context benchmark."""
import torch
from torch import nn

class MORPHModel(nn.Module):
    def __init__(s, G, emb_dim, dz=50):
        super().__init__()
        s.ce = nn.Sequential(nn.Linear(G, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        s.mu = nn.Linear(256, dz)
        s.pe = nn.Sequential(nn.Linear(emb_dim, 256), nn.ReLU(), nn.Linear(256, dz), nn.ReLU())
        s.gp = nn.Parameter(torch.randn(50, 50) * 0.02)
        jd = 2 * dz
        for i in [1, 2]:
            setattr(s, f'Wq{i}', nn.Linear(jd, jd))
            setattr(s, f'Wk{i}', nn.Linear(50, jd))
            setattr(s, f'Wv{i}', nn.Linear(50, jd))
            setattr(s, f'an{i}', nn.LayerNorm(jd))
            setattr(s, f'ff{i}', nn.Sequential(nn.Linear(jd, jd * 2), nn.ReLU(), nn.Linear(jd * 2, jd)))
            setattr(s, f'fn{i}', nn.LayerNorm(jd))
        s.dec = nn.Sequential(nn.Linear(jd, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, G))

    def _ca(s, z, Gp, i):
        Q = getattr(s, f'Wq{i}')(z).unsqueeze(1)
        K = getattr(s, f'Wk{i}')(Gp).unsqueeze(0).expand(Q.shape[0], -1, -1)
        V = getattr(s, f'Wv{i}')(Gp).unsqueeze(0).expand(Q.shape[0], -1, -1)
        a = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (Q.shape[-1] ** 0.5), dim=-1)
        z = getattr(s, f'an{i}')(z + torch.bmm(a, V).squeeze(1))
        return getattr(s, f'fn{i}')(z + getattr(s, f'ff{i}')(z))

    def forward(s, x, v):
        mu = s.mu(s.ce(x))
        zp = s.pe(v)
        z = torch.cat([mu, zp], -1)
        z = s._ca(z, s.gp, 1)
        z = s._ca(z, s.gp, 2)
        return s.dec(z)
