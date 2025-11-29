from __future__ import annotations
import torch
from typing import Any, List, Tuple, Optional

def _top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    _, preds = torch.max(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def train_core(device,model,train_loader,optimizer,criterion,use_amp=True,scaler=None):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        
        for imgs, targets,mask in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask=mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs,attn_mask=mask)
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
            for imgs, targets,masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                masks=masks.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(imgs,masks)
                    loss = criterion(logits, targets)
                bs = imgs.size(0)
                val_total += bs
                val_running_loss += loss.item() * bs
                val_running_acc  += _top1_accuracy(logits, targets) * bs

        last_val_acc = val_running_acc / val_total
        val_loss = val_running_loss / val_total
        
        return val_loss,last_val_acc

def traineval(epochs,device,model,train_loader,val_loader,criterion,optimizer,scheduler=None,use_amp=True,eval=True):
    best_val_acc = 0.0
    last_val_acc = 0.0
    model.to(device)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss,train_acc=train_core(device,model,train_loader,optimizer,criterion,use_amp)
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
