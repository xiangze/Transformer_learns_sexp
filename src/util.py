from __future__ import annotations
import torch
from typing import Any, List, Tuple, Optional
import numpy as np

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
        print("nan in ",d[k],d[k].shape)
        with open(f"nan_{k}.log","w") as fp:
            for k,v in d.items():
                pri(k,v,fp)
        exit()
    else:
        return None

def train_core(device,model,train_loader,optimizer,criterion,use_amp=True,scaler=None,debug=False):
        model.train()
        running_loss = 0.0
        total = 0
        for imgs, targets,mask,target_mask in train_loader:
            #print("img",imgs.shape,"mask",mask.shape,"target",targets.shape)
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask=mask.to(device, non_blocking=True)
            target_mask=target_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(imgs,attn_mask=mask)
                #loss = criterion(out, targets.to(out.dtype))
                # padding_mask: (B, L) True=PAD, False=# padding_mask: (B, L) True=PAD, False=有効
                valid_mask = (1-target_mask).unsqueeze(-1).float()  # (B,L,1)
                loss_raw = (out - targets) ** 2     # (B,L,D)
                loss = (loss_raw * valid_mask.squeeze(-1)).sum() / valid_mask.sum()
            if(debug):
                print("loss")
                nanindex({"img":imgs,"out":out,"img":imgs,"mask":mask},k="out")
                
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
        for imgs, targets,masks,target_mask in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks=masks.to(device, non_blocking=True)
            target_mask=target_mask.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(imgs,attn_mask=masks)
                #loss = criterion(logits, targets.to(logits.dtype))
                # padding_mask: (B, L) True=PAD, False=# padding_mask: (B, L) True=PAD, False=有効
                valid_mask = (1-target_mask).unsqueeze(-1).float()  # (B,L,1)
                loss_raw = (out-targets)**2     # (B,L,D)
                loss = (loss_raw*valid_mask.squeeze(-1)).sum() / valid_mask.sum()

            bs = imgs.size(0)
            val_total += bs
            val_running_loss += loss.item() * bs

    val_loss = val_running_loss / val_total
    return val_loss 

def traineval(epochs,device,model,train_loader,val_loader,criterion,optimizer,scheduler=None,use_amp=True,eval=True,peri=100,debug=False):
    best_val_loss = 1e10
    val_loss=0.
    model.to(device)
    scaler=torch.cuda.amp.GradScaler()
    for epoch in range(1, epochs + 1):
        train_loss=train_core(device,model,train_loader,optimizer,criterion,use_amp,scaler,debug)
        
        if(epoch%peri==0):
            msg=f"[{epoch:03d}/{epochs}] "+f"train_loss={train_loss:.4f} "
            if(eval):
                val_loss=eval_core(device,model,val_loader,criterion,use_amp)
                msg+=f"val_loss={val_loss:.4f}  "
            if(scheduler!=None):
                msg+=f"lr={scheduler.get_last_lr()[0]:.6f}"
            print(msg)

        if(scheduler!=None):
            scheduler.step()

        best_val_loss= min(best_val_loss, val_loss)            
    last_val_loss=val_loss
    return best_val_loss,last_val_loss
