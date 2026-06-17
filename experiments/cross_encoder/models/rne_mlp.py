"""RNE + MLP Decoder — Cross-Encoder改造。支持L1/L1Tilde切换。"""
import time, numpy as np
import torch, torch.nn as nn
from models.basemodel import BaseModel

class RNEMLP(BaseModel):
    def __init__(self, num_nodes, embed_size, max_distance=1.0, parts=None,
                 use_l1tilde=False, l1tilde_r=62, l1tilde_s=2):
        super().__init__()
        self.embed_size, self.max_distance = embed_size, max_distance
        self.use_l1tilde = use_l1tilde
        self.l1tilde_r, self.l1tilde_s = l1tilde_r, l1tilde_s
        self.parts = torch.from_numpy(parts) if parts is not None else None
        if use_l1tilde:
            assert l1tilde_r + l1tilde_s <= embed_size

        self.embedding = nn.Embedding(num_nodes, embed_size)
        nn.init.uniform_(self.embedding.weight, -3/2, 3/2)

        # MLP: concat→y_o,y_d
        d_in = embed_size * 2
        hidden = 512
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, d_in),
        )

    def forward(self, x1, x2):
        e1, e2 = self.embedding(x1), self.embedding(x2)
        out = self.mlp(torch.cat([e1, e2], dim=1))
        y_o, y_d = torch.chunk(out, 2, dim=1)
        if self.use_l1tilde:
            r, s = self.l1tilde_r, self.l1tilde_s
            sym = torch.abs(y_d[:, :r] - y_o[:, :r]).sum(dim=1, keepdim=True)
            asym = (y_d[:, r:r + s] - y_o[:, r:r + s]).sum(dim=1, keepdim=True)
            return (sym + asym) * self.max_distance
        return torch.norm(y_d - y_o, p=1, dim=1, keepdim=True) * self.max_distance

    def _train_step(self, x1, x2, y, criterion, optimizer):
        optimizer.zero_grad()
        yp = self.forward(x1, x2) / self.max_distance
        loss = criterion(yp, y / self.max_distance)
        loss.backward(); optimizer.step(); return loss

    def fit(self, dataloader, criterion, optimizer, val_dataloader=None,
            epochs=1, display_step=10, device="cpu", time_limit=None,
            fast_dev_run=False, **kwargs):
        self.train(); self.to(device); criterion.to(device)
        loss_ep, loss_it, val_mre, time_h = [], [], [], []
        display_step = max(1, len(dataloader) // display_step)
        st = time.perf_counter()

        if self.parts is not None:
            nl = self.parts.shape[1]
            he = [5]*(nl-1) + [10]; self.parts = self.parts.to(device)
            prev = None
            for lv in range(nl):
                pi = self.parts[:, lv]
                if lv > 0:
                    with torch.no_grad():
                        self.embedding.weight.data[pi] = prev[self.parts[:, lv-1]]
                for ep in range(he[lv]):
                    for batch in dataloader:
                        i, j, d = batch[0], batch[1], batch[2].unsqueeze(-1)
                        i, j, d = pi[i].to(device), pi[j].to(device), d.to(device)
                        self._train_step(i, j, d, criterion, optimizer)
                prev = self.embedding.weight.data.clone()

        tlc = None
        for epoch in range(epochs):
            rl = 0.0
            for batch in dataloader:
                i, j, d = batch[0], batch[1], batch[2].unsqueeze(-1)
                i, j, d = i.to(device), j.to(device), d.to(device)
                loss = self._train_step(i, j, d, criterion, optimizer)
                rl += loss.item(); loss_it.append(loss.item())
            avg = rl / len(dataloader); loss_ep.append(avg)
            el = (time.perf_counter() - st) / 60; time_h.append(el)
            vs = ""
            if val_dataloader:
                vp, vt, _ = self.evaluate(val_dataloader, device=device, verbose=False, profile_time=False)
                vm = float(np.mean(np.abs(vp-vt)/np.maximum(vt,1e-6)))
                val_mre.append(vm); vs = f", Val MRE: {vm:.2%}"; self.train()
            print(f"Epoch: {epoch+1:>2}/{epochs}, Time: {el:.1f}min, Loss: {avg:.6f}{vs}")
            if time_limit and el >= time_limit and tlc is None:
                tlc = {"epoch": epoch+1, "time_min": el, "val_mre": val_mre[-1] if val_mre else None}
        r = {"loss_epoch_history": loss_ep, "loss_iter_history": loss_it,
             "val_mre_epoch_history": val_mre, "time_history": time_h}
        if tlc: r["time_limit_checkpoint"] = tlc
        return r
