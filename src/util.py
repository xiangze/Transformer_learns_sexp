from __future__ import annotations
import torch
from typing import Any, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

def make_train_eval_loader(device,name,batch_size, tfm_train,tfm_eval, custom_num_classes=None,num_workers=1,
                           data_root="data/", download=False):
    train_ds, n_tr = _make_dataset(name, data_root, train=True,  download=download, tfm=tfm_train)
    val_ds,   n_va = _make_dataset(name, data_root, train=False, download=download, tfm=tfm_eval)
    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    num_classes = custom_num_classes if custom_num_classes is not None else max(n_tr, n_va)
    return train_loader,val_loader,num_classes

def smart_getattr(mod, candidates: Sequence[str]) -> Optional[Callable]:
    """Return first existing attribute (callable) by trying candidates in order."""
    for name in candidates:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            if callable(fn):
                return fn
    return None

def fail_with_attributes(mod, purpose: str) -> None:
    attrs = [a for a in dir(mod) if not a.startswith("_")]
    raise AttributeError(
        f"Cannot find a suitable function for {purpose} in module {mod.__name__}.\n"
        f"Available attributes:\n- " + "\n- ".join(attrs)
    )

def train_core(device,model,train_loader,optimizer,criterion,use_amp=True,scaler=None):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, targets)
            
            if(use_amp and scaler !=None):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            total += bs
            running_loss += loss.item() * bs
            running_acc  += _top1_accuracy(logits.detach(), targets) * bs

        train_loss = running_loss / total
        train_acc  = running_acc / total
        return train_loss,train_acc

def eval_core(device,model,val_loader,criterion,use_amp=True):
        # ---- eval ----
        model.eval()
        val_total = 0
        val_running_acc = 0.0
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, targets)
                bs = imgs.size(0)
                val_total += bs
                val_running_loss += loss.item() * bs
                val_running_acc  += _top1_accuracy(logits, targets) * bs

        last_val_acc = val_running_acc / val_total
        val_loss = val_running_loss / val_total
        
        return val_loss,last_val_acc

def traineval(epochs,device,model,train_loader,val_loader,criterion,optimizer,scheduler=None,use_amp=True,scaler=None,eval=True):
    best_val_acc = 0.0
    last_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss,train_acc=train_core(device,model,train_loader,optimizer,criterion,use_amp,scaler)
        if(eval):
            val_loss,last_val_acc=eval_core(device,model,val_loader,criterion,use_amp)

        msg=f"[{epoch:03d}/{epochs}] "+f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "

        if(eval):
            msg+=+f"val_loss={val_loss:.4f} acc={last_val_acc:.4f} | "
        if(scheduler!=None):
            msg+=f"lr={scheduler.get_last_lr()[0]:.6f}"
    
        print(msg)

        if(scheduler!=None):
            scheduler.step()

        best_val_acc = max(best_val_acc, last_val_acc)            
    return best_val_acc,last_val_acc
