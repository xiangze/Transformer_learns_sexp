from __future__ import annotations
import torch
from dataclasses import dataclass, asdict, field
from typing import Any, List, Tuple, Optional
import numpy as np
import torch.nn.functional as F
import mlflow
import mlflow.pytorch

def mprint(msg,on):
    if(on):
        print(msg)

def dprint(s,fp,on=True):
    if(on):
        print(s)
        if(type(fp)==list):
            for f in fp:
                print(s,file=f)
        else:   
            print(s,file=fp)

def pri(k,v,fp):
    print(k,f"shape:{v.shape}",file=fp)
    if(v.dim()>2):
        for i in range(v.shape[0]):
            np.savetxt(fp,v[i].cpu().detach().numpy())
    else:
        np.savetxt(fp,v.cpu().detach().numpy())        

def nanindex(d,k="out") -> torch.Tensor:
    """テンソル内のNaNのインデックスを返す"""
    nanidx = torch.isnan(d[k]).nonzero(as_tuple=False)
    if(nanidx.shape[0]>0):
        print("loss")
        print("nan in ",d[k],d[k].shape)
        with open(f"nan_{k}.log","w") as fp:
            for k,v in d.items():
                pri(k,v,fp)
        exit()
    else:
        return None


@dataclass
class MLflowLogger:
    experiment_name: str = "default"
    run_name: Optional[str] = None
    tracking_uri: str = "./mlruns"          # ローカル保存先（変更可）
    params: dict = field(default_factory=dict)
    _run_id: Optional[str] = field(default=None, init=False, repr=False)

    def start(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_name=self.run_name)
        self._run_id = run.info.run_id
        if self.params:
            mlflow.log_params(self.params)
        return self

    def log(self, metrics: dict, step: int):
        """毎エポック呼ぶ。metrics = {"train_loss": 0.1, "val_loss": 0.2, ...}"""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: nn.Module, artifact_path: str = "model"):
        """学習終了後にモデルを保存する"""
        mlflow.pytorch.log_model(model, artifact_path)

    def end(self):
        mlflow.end_run()

    # context manager として使えるようにする
    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.end()

def compute_loss(out, targets, target_mask, task="regression"):
    if task == "classification":
        # out: (B, num_classes), targets: (B,) の整数ラベル
        valid = ~target_mask                          # (B,)
        loss = F.cross_entropy(out[valid], targets[valid].long())
    else:
        valid_mask = (~target_mask).unsqueeze(-1).float()  # (B,L,1)
        loss_raw = (out - targets) ** 2
        loss = (loss_raw * valid_mask.squeeze(-1)).sum() / valid_mask.sum()
    return loss

def train_core(device, model, train_loader, optimizer, criterion, use_amp=True, scaler=None, debug=False, task="regression"):
    model.train()
    running_loss = 0.0
    total = 0
    for i, data in enumerate(train_loader):
        imgs        = data[0][:, 0].to(device, non_blocking=True)
        targets     = data[0][:, 1].to(device, non_blocking=True)
        mask        = data[0][:, 2].to(device, non_blocking=True, dtype=torch.bool)
        target_mask = data[0][:, 3].to(device, non_blocking=True, dtype=torch.bool)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out  = model(imgs, attn_mask=mask)
            loss = compute_loss(out, targets, target_mask, task)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        total        += bs
        running_loss += loss.item() * bs

    return running_loss / total


def eval_core(device, model, val_loader, criterion, use_amp=True, task="regression"):
    model.eval()
    val_total = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            imgs        = data[0][:, 0].to(device, non_blocking=True)
            targets     = data[0][:, 1].to(device, non_blocking=True)
            masks       = data[0][:, 2].to(device, non_blocking=True, dtype=torch.bool)
            target_mask = data[0][:, 3].to(device, non_blocking=True, dtype=torch.bool)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out  = model(imgs, attn_mask=masks)
                loss = compute_loss(out, targets, target_mask, task)

            bs = imgs.size(0)
            val_total        += bs
            val_running_loss += loss.item() * bs

    return val_running_loss / val_total


def traineval(epochs, device, model, train_loader, val_loader, criterion, optimizer,
              scheduler=None, use_amp=True, eval=True, peri=10, debug=False,
              logger: Optional[MLflowLogger] = None,
              fpw=None, task="regression"):
    best_val_loss = 1e10
    val_loss = 0.
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, epochs + 1):
        train_loss = train_core(device, model, train_loader, optimizer, criterion, use_amp, scaler, debug, task)
        if epoch % peri == 0:
            metrics = {"train_loss": train_loss}
            if eval:
                val_loss = eval_core(device, model, val_loader, criterion, use_amp, task)
                metrics["val_loss"] = val_loss

                if logger:
                    logger.log(metrics, step=epoch)
                
            msg += f"[{epoch:03d}/{epochs}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())                
            dprint(msg, fpw)
        if scheduler is not None:
            scheduler.step()
        best_val_loss = min(best_val_loss, val_loss)
    return train_loss, best_val_loss, val_loss

