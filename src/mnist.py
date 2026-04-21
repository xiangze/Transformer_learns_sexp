from __future__ import annotations
import util
import pipeline_cv_train as p
import attentiononly as atn
import transformer_dick_fixed_embed as fixed
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch import tensor,save
import itertools
import numpy as np

IMAGE_DATASETS = {
    "mnist":         ("MNIST",        1, 28, 28,  10),
    "fashion_mnist": ("FashionMNIST", 1, 28, 28,  10),
    "cifar10":       ("CIFAR10",      3, 32, 32,  10),
    "cifar100":      ("CIFAR100",     3, 32, 32, 100),
}

LABEL_NAMES = {
    "mnist":         list(map(str, range(10))),
    "fashion_mnist": ["T-shirt","Trouser","Pullover","Dress","Coat",
                      "Sandal","Shirt","Sneaker","Bag","Ankle boot"],
    "cifar10":       ["airplane","automobile","bird","cat","deer",
                      "dog","frog","horse","ship","truck"],
    "cifar100":      [f"c{i}" for i in range(100)],
}

# タスク別のデフォルト N, M
TASK_DEFAULTS = {
    "mnist":         (28, 28),   # 784 px → 28 行×28 列
    "fashion_mnist": (28, 28),
    "cifar10":       (32, 96),   # 3072 px → 32×96
    "cifar100":      (32, 96),
    "sin":           (16,  7),
    "csv":           (16,  7),
}

def load_image_dataset(name, data_root="./data", max_train=None, max_test=None,to1D=True):
    ds_name, C, H, W, n_classes = IMAGE_DATASETS[name]
    img_flat = C * H * W

    if C == 1:
        mean, std = (0.5,), (0.5,)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)

    # 学習用: CIFAR は RandomHorizontalFlip を追加
    aug = [transforms.RandomHorizontalFlip()] if C == 3 else []
    if(to1D):
        tf_train = transforms.Compose(aug + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Lambda(lambda x: x.reshape(-1)),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Lambda(lambda x: x.reshape(-1)),
        ])
    else:
        tf_train = transforms.Compose(aug + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    ds_cls   = getattr(torchvision.datasets, ds_name)
    train_ds = ds_cls(root=data_root, train=True,  download=True, transform=tf_train)
    test_ds  = ds_cls(root=data_root, train=False, download=True, transform=tf_test)

    if max_train is not None:
        train_ds = Subset(train_ds, list(range(min(max_train, len(train_ds)))))
    if max_test is not None:
        test_ds  = Subset(test_ds,  list(range(min(max_test,  len(test_ds)))))

    return train_ds, test_ds, n_classes, img_flat

def embed_samples(model, X: torch.Tensor) -> torch.Tensor:
    """
    画像モデルなら embed 層を通して (n, N, M) に変換する。
    """
    model.eval()
    with torch.no_grad():
        emb = model.embed(X.float())
        return emb.reshape(-1, model.n_input, model.n_seq)

def parse_args():
    p = argparse.ArgumentParser(description=" Attention only vs Attention+FNN performence test by MNIST")
    p.add_argument("--task", type=str, default="mnist", choices=["mnist","fashion_mnist","cifar10","cifar100"])
    p.add_argument("--data_root",  type=str, default="./dataset")
    p.add_argument("--max_train",  type=int, default=None, help="学習サンプル上限 (None=全件)")
    p.add_argument("--max_test",   type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--max_len",    type=int, default=9999, help="max length of input sequence")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--kfold", type=int, default=1, help="交差検証のfold数")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dropout", type=float, default=0.2, help="dropout")
    p.add_argument("--show_msg", action="store_false")
    return p.parse_args()

def train_one_fold(args,train_ds,test_ds, params_tr,pname,device,fpw=None,k=0):
    model=p.make_model(params_tr,"fixed",params_tr["vocab_size"],args.type,args.debug).to(device)
    if(k==0):
        print("paramsize ",sum(pp.numel() for pp in model.parameters()),file=fpw)
    modelname=f"model/{pname}_{k}.pth"
    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=1, pin_memory=pin)
    val_loader   = DataLoader(test_ds , batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=pin)

    criterion = torch.nn.CrossEntropyLoss()
    opt=torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.95 ** epoch)
    train_loss,best_val_loss,last_val_loss=util.traineval(args.epochs,device,model,train_loader,val_loader,criterion,opt,scheduler,
                                                          use_amp=args.use_amp,eval=True,peri=max(1,args.epochs//10),debug=args.debug,fpw=fpw,task=args.task)

    msg=f"{k+1}/{args.kfold}] fold, train loss: {train_loss}, best val loss: {best_val_loss}, last val loss: {last_val_loss}"
    print(msg,file=fpw)
    save(model.state_dict(), modelname)

def test1(args,params_tr,train_ds_all, test_ds_all,fpw):
    print(params_tr,file=fpw)
    pname = p.makesuf(args,params_tr,{})
    if(args.kfold>1):
        folds = p.kfold_split(len(train_ds_all), args.kfold, args.seed)[:1]
        for k, (tr_idx, va_idx) in enumerate(folds):
            train_ds = tensor(np.array([train_ds_all[i] for i in tr_idx]))
            test_ds   = tensor(np.array([test_ds_all[i] for i in va_idx]))
            train_one_fold(args,train_ds,test_ds,params_tr,pname,args.device,fpw,k)
    else:   
            #mask?
            train_one_fold(args,train_ds_all,test_ds_all,params_tr,pname,args.device,fpw,0)
        
def test(args,task,to1D=True):
    args.type="classification"
    train_ds_all, test_ds_all, n_classes, img_flat = load_image_dataset(task,
                                            data_root=args.data_root,
                                            max_train=args.max_train, max_test=args.max_test,to1D=to1D)
    print(f"train={len(train_ds_all)}, test={len(test_ds_all)}, classes={n_classes}, img_flat(C*H*W)={img_flat}")
    # input input_ids,mask
    # pos_ids = torch.arange(L).unsqueeze(0).expand(input_ids.shape)
    # tok = nn.Embedding(vocab_size, d_model)
    # pos = nn.Embedding(max_len, d_model)
    # x = tok(input_ids) + pos(pos_ids)
    args.n_classes=n_classes
    args.max_len=img_flat
    vocab_size=256 #for embedded layer

    with open(f"log/vs_{args.task}.log","w") as fpw:
        for ato,rec,d_model, head, layer, dimff in itertools.product(
            [False,True],[False,True],[256],[8],[3],[256]):
            params_tr ={"d_model":d_model, "nhead":head, "num_layer" :layer, "dim_ff": dimff, "max_len": args.max_len, "vocab_size":vocab_size,
                        "model":"fixed","attentiononly":ato,"recursive":rec,"noembedded":True,"activate":False,"dropout":args.dropout}
            test1(args,params_tr,train_ds_all, test_ds_all,fpw)

def test_mnist(args):
    test(args,"mnist")

if __name__ == "__main__":
    args   = parse_args()
    test_mnist(args)

    

