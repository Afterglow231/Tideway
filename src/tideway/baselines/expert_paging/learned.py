"""Per-layer FlashMoE predictor training with held-out calibration validation."""

import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .policies import History


def network(experts):
    return torch.nn.Sequential(
        torch.nn.Linear(2 * experts, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, experts),
    )


def samples(sequences, experts, cap, horizon=128):
    features, targets, masks = [], [], []
    for seq in sequences:
        seq = np.asarray(seq, np.int64)
        future = np.full(experts, len(seq) + horizon, np.int64)
        distance = np.empty((len(seq), experts), np.float32)
        for t in range(len(seq) - 1, -1, -1):
            distance[t] = np.minimum(future - t, horizon) / horizon
            future[seq[t]] = t
        history = History(1, experts)
        resident = set()
        for t, ids in enumerate(seq):
            history.observe(0, ids[None, :])
            active = set(map(int, ids))
            candidates = resident - active
            if candidates:
                mask = np.zeros(experts, np.float32)
                mask[list(candidates)] = 1
                features.append(history.features(0))
                targets.append(distance[t])
                masks.append(mask)
            for e in ids:
                e = int(e)
                if e not in resident:
                    if len(resident) >= cap:
                        eligible = resident - active
                        if not eligible:
                            raise ValueError("cap smaller than active set")
                        resident.remove(max(eligible, key=lambda x: distance[t, x]))
                    resident.add(e)
    return tuple(torch.from_numpy(np.asarray(a, np.float32)) for a in (features, targets, masks))


def train(routes_path, out, experts=128, layers=30, cap=60, max_epochs=200):
    routes_path = Path(routes_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    raw = routes_path.read_bytes()
    records = json.loads(raw)
    names = list(dict.fromkeys(r["case"] for r in records))
    if len(names) < 3:
        raise ValueError("Need separate calibration training and validation requests")
    models = []
    reports = []
    torch.manual_seed(202709)
    for layer in range(layers):
        sequences = {name: [] for name in names}
        for r in records:
            if r["layer"] == layer:
                sequences[r["case"]].extend(r["ids"])
        x, y, m = samples([sequences[n] for n in names[:-1]], experts, cap)
        xv, yv, mv = samples([sequences[names[-1]]], experts, cap)
        model = network(experts)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # The paper's "learning rate decay=1e-2" is ambiguous. Fix a declared
        # final/start LR ratio of .01 over the maximum budget; never tune on eval.
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.01 ** (1 / max_epochs)
        )
        best = float("inf")
        saved = None
        stale = 0
        losses = []
        for epoch in range(max_epochs):
            model.train()
            for ix in torch.randperm(len(x)).split(128):
                optimizer.zero_grad(set_to_none=True)
                loss = (((model(x[ix]) - y[ix]) ** 2) * m[ix]).sum() / m[ix].sum().clamp_min(1)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val = float((((model(xv) - yv) ** 2) * mv).sum() / mv.sum().clamp_min(1))
            losses.append(val)
            scheduler.step()
            if val < best - 1e-6:
                best = val
                saved = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 10:
                break
        model.load_state_dict(saved)
        models.append(saved)
        report = dict(
            layer=layer,
            train_states=len(x),
            validation_states=len(xv),
            epochs=len(losses),
            best_validation_mse=best,
            validation_curve=losses,
        )
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != "validation_curve"}), flush=True)
    torch.save(dict(experts=experts, layers=layers, states=models), out / "predictor.pt")
    manifest = dict(
        source_sha256=hashlib.sha256(raw).hexdigest(),
        training_cases=names[:-1],
        validation_cases=names[-1:],
        evaluation_access=False,
        architecture=[2 * experts, 128, 128, experts],
        activation="SiLU",
        loss="MSE masked to Belady-resident eligible victims",
        label="next-use distance clipped to128 and divided by128; unavailable future beyond request treated as horizon",
        optimizer="AdamW lr=.001 weight_decay=.01",
        lr_schedule="exponential final/start=.01",
        max_epochs=max_epochs,
        patience=10,
        adaptation="small independent code-calibration pilot, not original 512 TriviaQA samples; no evaluation tuning",
        layers=reports,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class Predictor:
    def __init__(self, path):
        data = torch.load(Path(path) / "predictor.pt", map_location="cpu", weights_only=True)
        self.experts, self.layers = int(data["experts"]), int(data["layers"])
        self.models = []
        for state in data["states"]:
            model = network(data["experts"])
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

    def __call__(self, layer, features):
        with torch.no_grad():
            return self.models[layer](torch.from_numpy(features)).numpy()
