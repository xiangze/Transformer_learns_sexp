from __future__ import annotations
import torch
from typing import Any, List, Tuple, Optional

def _top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    #_, preds = torch.max(logits, dim=1)
    preds = logits.argmax(dim=1)
    print("logits",logits.shape,"preds",preds.shape,"targets",targets.shape)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def train_core(device,model,train_loader,optimizer,criterion,use_amp=True,scaler=None):
        model.train()
        running_loss = 0.0
        total = 0
        for imgs, targets,mask in train_loader:
            #print("img",imgs.shape,"mask",mask.shape,"target",targets.shape)
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask=mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(imgs,attn_mask=mask)
                loss = criterion(out, targets.to(out.dtype))
            
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

        train_loss = running_loss / total
        return train_loss 

def eval_core(device,model,val_loader,criterion,use_amp=True):
    model.eval()
    val_total = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for imgs, targets,masks in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks=masks.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs,attn_mask=masks)
                loss = criterion(logits, targets.to(logits.dtype))
            bs = imgs.size(0)
            val_total += bs
            val_running_loss += loss.item() * bs

    val_loss = val_running_loss / val_total
    return val_loss 

def traineval(epochs,device,model,train_loader,val_loader,criterion,optimizer,scheduler=None,use_amp=True,eval=True,peri=100):
    best_val_loss = 0.0
    val_loss=0.
    model.to(device)
    scaler=torch.cuda.amp.GradScaler()
    for epoch in range(1, epochs + 1):
        train_loss=train_core(device,model,train_loader,optimizer,criterion,use_amp,scaler)
        msg=f"[{epoch:03d}/{epochs}] "+f"train_loss={train_loss:.4f} "

        if(epoch%peri==0):
            if(eval):
                val_loss=eval_core(device,model,val_loader,criterion,use_amp)
                msg+=f"val_loss={val_loss:.4f}  "
            if(scheduler!=None):
                msg+=f"lr={scheduler.get_last_lr()[0]:.6f}"

            print(msg)

        if(scheduler!=None):
            scheduler.step()

        best_val_loss= max(best_val_loss, val_loss)            
    last_val_loss=val_loss
    return best_val_loss,last_val_loss
