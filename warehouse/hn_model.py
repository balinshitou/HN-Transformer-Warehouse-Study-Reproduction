"""Published Transformer backbone and inference helpers from src/models.py and src/pkg_eval.py."""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalNorm(nn.Module):
    """Apply fixed training-set normalization with shared horizontal scales."""
    # The fallback constants support checkpoints that omit training-set statistics.
    def __init__(self, enabled=True, mu=None, sd=None):
        super().__init__()
        self.enabled = enabled
        if not enabled:
            return
        # Channel order: [gx, gy, gz, ax, ay, az]. Share horizontal scales for yaw equivariance.
        if mu is None:
            mu = [0., 0., 0., 0., 0., 9.810]
        if sd is None:
            sd = [0.854, 0.854, 0.993, 2.134, 2.134, 2.618]
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))

    def forward(self, x):
        if not self.enabled:
            return x
        return (x - self.mu.view(1, -1, 1)) / self.sd.view(1, -1, 1)


class HeadingNorm(nn.Module):
    """Align the horizontal acceleration principal axis with the x-axis.

    The input is already gravity-aligned. The forward pass returns the yaw
    angle so that predicted velocity can be rotated back to the input frame."""
    def forward(self, x):
        # x has shape [B, 6, T]: three gyroscope and three gravity-aligned accelerometer channels.
        a = x[:, 3:5, :]                                   # [B,2,T]
        ac = a - a.mean(dim=2, keepdim=True)
        C = torch.einsum('bit,bjt->bij', ac, ac) / ac.shape[2]
        # Closed-form principal eigenvector angle for a symmetric 2 x 2 matrix.
        cxx, cyy, cxy = C[:, 0, 0], C[:, 1, 1], C[:, 0, 1]
        theta = 0.5 * torch.atan2(2 * cxy, cxx - cyy)      # Principal-axis angle
        # The principal axis has a 180-degree ambiguity. Use the sign of the
        # third moment of projected acceleration to choose its orientation.
        cth, sth = torch.cos(theta), torch.sin(theta)
        proj = ac[:, 0, :] * cth[:, None] + ac[:, 1, :] * sth[:, None]
        skew = (proj ** 3).mean(dim=1)
        theta = theta + torch.where(skew < 0, torch.pi, 0.0)
        c, s = torch.cos(-theta), torch.sin(-theta)        # Align the principal axis with the x-axis.
        out = x.clone()
        for off in (0, 3):
            xx, yy = x[:, off, :], x[:, off + 1, :]
            out[:, off, :] = c[:, None] * xx - s[:, None] * yy
            out[:, off + 1, :] = s[:, None] * xx + c[:, None] * yy
        return out, theta


class PreFilter(nn.Module):
    """Expand six input channels into raw, low-, band-, and high-pass components.

    The fixed Gaussian filters separate components below approximately
    0.5 Hz, between 0.5 and 3 Hz, and above approximately 3 Hz."""
    def __init__(self, rate=200.0, enabled=True, gnorm=False):
        super().__init__()
        self.enabled = enabled
        self.gn = GlobalNorm(enabled=gnorm)
        if not enabled:
            self.out_mult = 1
            return
        self.out_mult = 4
        def gauss(sigma):
            r = max(1, int(3 * sigma))
            x = torch.arange(-r, r + 1, dtype=torch.float32)
            k = torch.exp(-0.5 * (x / sigma) ** 2)
            return (k / k.sum()).view(1, 1, -1)
        self.register_buffer("k_lo", gauss(rate / (2 * math.pi * 0.5)))
        self.register_buffer("k_mid", gauss(rate / (2 * math.pi * 3.0)))

    def forward(self, x):
        x = self.gn(x)
        if not self.enabled:
            return x
        B, C, T = x.shape
        z = x.reshape(B * C, 1, T)
        lo = F.conv1d(z, self.k_lo, padding=self.k_lo.shape[-1] // 2)[..., :T]
        mid = F.conv1d(z, self.k_mid, padding=self.k_mid.shape[-1] // 2)[..., :T]
        lo = lo.reshape(B, C, T); mid = mid.reshape(B, C, T)
        return torch.cat([x, lo, mid - lo, x - mid], dim=1)


def _unrotate(v, theta):
    """Rotate the first two output channels back to the original frame."""
    if theta is None:
        return v
    c, s = torch.cos(theta), torch.sin(theta)
    out = torch.stack([c * v[:, 0] - s * v[:, 1],
                       s * v[:, 0] + c * v[:, 1]], dim=1)
    return out if v.shape[1] == 2 else torch.cat([out, v[:, 2:]], dim=1)


class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        # For odd d, the sine columns outnumber the cosine columns by one.
        pe[:, 0::2] = torch.sin(pos * div[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.shape[1]]


class TransformerNet(nn.Module):
    """Embed temporal patches and regress velocity from the Transformer CLS token."""
    def __init__(self, cin=6, d=128, nhead=4, nlayer=4, pre=True, patch=8,
                 hnorm=False, nout=2, rate=200.0):
        super().__init__()
        self.nout = nout
        self.hn = HeadingNorm() if hnorm else None
        # Use the checkpoint sampling rate to construct the fixed filter bank.
        self.pre = PreFilter(rate=rate, enabled=pre)
        self.embed = nn.Conv1d(cin * self.pre.out_mult, d, patch, stride=patch)
        self.pos = PosEnc(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.1,
                                         batch_first=True, activation="gelu",
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, nlayer)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, nout))

    def forward(self, x):
        th = None
        if self.hn is not None:
            x, th = self.hn(x)
        h = self.embed(self.pre(x)).transpose(1, 2)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1)
        h = self.enc(self.pos(h))
        return _unrotate(self.head(h[:, 0]), th)


def pick_device(verbose=True):
    """Select the compute device specified by HN_DEVICE. Default to CPU."""
    import torch
    requested = os.environ.get("HN_DEVICE", "cpu").strip().lower()
    if requested not in {"cpu", "cuda", "mps", "auto"}:
        raise ValueError("HN_DEVICE must be cpu, cuda, mps, or auto.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("HN_DEVICE is cuda, but PyTorch cannot access CUDA.")
    if requested == "mps" and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("HN_DEVICE is mps, but PyTorch cannot access Apple MPS.")
    use_cuda = requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
    use_mps = requested == "mps" or (
        requested == "auto" and not use_cuda and
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if use_cuda:
        dev = torch.device("cuda"); nm = torch.cuda.get_device_name(0)
    elif use_mps:
        dev = torch.device("mps"); nm = "Apple MPS"
    else:
        dev = torch.device("cpu")
        n = os.cpu_count() or 4
        # Cap the thread count at the number of logical CPUs.
        torch.set_num_threads(max(1, min(n, int(os.environ.get("PKG_THREADS", n)))))
        nm = f"CPU ({torch.get_num_threads()} threads)"
    if verbose:
        print(f"[Device] {nm}", flush=True)
    return dev


def predict(net, feat, dev, win_sec, rate, chunk=2048):
    ds = int(round(200.0 / rate)); w = int(round(win_sec * rate))
    f = feat[::ds]
    ids = np.arange(0, len(f) - w, max(1, int(round(rate * 0.05))), dtype=int)
    if len(ids) < 2:
        return None, None
    out = np.empty((len(ids), 2), np.float32)
    with torch.no_grad():
        for i0 in range(0, len(ids), chunk):
            idx = ids[i0:i0 + chunk]
            x = np.stack([f[j:j + w].T for j in idx], 0).astype(np.float32)
            out[i0:i0 + len(idx)] = net(torch.from_numpy(x).to(dev)).cpu().numpy()[:, :2]
    return ids * ds, np.asarray(out, float)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint['arch'] != 'transformer':
        raise ValueError('This package requires the Transformer checkpoint used in the manuscript.')
    width = float(checkpoint.get('width', 1.0))
    dimension = int(round(128 * width))
    heads = max(2, dimension // 32)
    if dimension % heads:
        dimension += heads - dimension % heads
    rate = float(checkpoint.get('rate', 200.0))
    model = TransformerNet(d=dimension, nhead=heads, pre=checkpoint.get('pre', True),
                           hnorm=checkpoint.get('hnorm', False),
                           nout=checkpoint.get('nout', 2), rate=rate)
    if checkpoint.get('gnorm'):
        model.pre.gn = GlobalNorm(enabled=True, mu=checkpoint.get('gnorm_mu'),
                                  sd=checkpoint.get('gnorm_sd'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval().to(device), float(checkpoint.get('win_sec', 1.0)), rate
