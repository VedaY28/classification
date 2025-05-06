import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import time
import matplotlib.pyplot as plt

digits = (
    "data/digitdata/trainingimages",
    "data/digitdata/traininglabels",
    "data/digitdata/validationimages",
    "data/digitdata/validationlabels",
    "data/digitdata/testimages",
    "data/digitdata/testlabels"
)
face = (
    "data/facedata/facedatatrain",
    "data/facedata/facedatatrainlabels",
    "data/facedata/facedatavalidation",
    "data/facedata/facedatavalidationlabels",
    "data/facedata/facedatatest",
    "data/facedata/facedatatestlabels"
)

CHAR_TO_PIXEL = {' ':0, '+':1, '#':1}

def parse_images(image_file, label_file, H, W):
    labels = []
    with open(label_file) as lf:
        for l in lf:
            l = l.strip()
            if l:
                labels.append(int(l))
    lines = [ln.rstrip('\n') for ln in open(image_file)]
    imgs = []
    N = len(labels)
    for i in range(N):
        block = lines[i*H:(i+1)*H]
        flat = []
        for row in block:
            row = row.ljust(W)
            for ch in row:
                flat.append(CHAR_TO_PIXEL.get(ch,0))
        imgs.append(flat)
    return torch.tensor(imgs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

class TorchNN(nn.Module):
    def __init__(self, inp, h1, h2, out):
        super().__init__()
        self.l1 = nn.Linear(inp,h1)
        self.l2 = nn.Linear(h1,h2)
        self.l3 = nn.Linear(h2,out)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)        

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1)
        return (pred==y).float().mean().item()

def train(model, Xtr, ytr, Xval, yval, epochs=5, lr=1e-2):
    best = copy.deepcopy(model)
    best_acc = 0.
    opt = optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for i in range(len(Xtr)):
            x =       Xtr[i].unsqueeze(0)
            target =  ytr[i].unsqueeze(0)
            opt.zero_grad()
            out =      model(x)
            loss =     crit(out,target)
            loss.backward()
            opt.step()
        acc = evaluate(model, Xval, yval)
        if acc>best_acc:
            best_acc, best = acc, copy.deepcopy(model)
    return best

def run(img_f, lbl_f, vimg_f, vlbl_f,  \
        timg_f, tlbl_f, classes, H, W, title):
    Xtr, ytr = parse_images(img_f,  lbl_f,  H,W)
    Xval, yval = parse_images(vimg_f, vlbl_f,H,W)
    Xte, yte  = parse_images(timg_f, tlbl_f, H,W)
    inp = H*W
    stats=[]
    print(f"\n{title}:")
    for pct in range(10,101,10):
        accs=[]
        t0 = time.time()
        N=len(Xtr)
        for _ in range(5):
            idx = list(range(N))
            random.shuffle(idx)
            k = int(pct/100*N)
            sel = idx[:k]
            Xsub, ysub = Xtr[sel], ytr[sel]
            net = TorchNN(inp, 64,16, classes)
            best = train(net, Xsub, ysub, Xval, yval, epochs=5, lr=0.01)
            accs.append(evaluate(best, Xte, yte))
        dt = time.time()-t0
        mean, std = np.mean(accs), np.std(accs)
        stats.append((pct,mean,std,dt))
        print(f"{pct}% → Acc {mean:.4f}±{std:.4f}  Time {dt:.2f}s")
    return stats

if __name__=='__main__':
    stats_d = run(*digits, classes=10, H=28, W=28, title="Digits")
    stats_f = run(*face,   classes=2,  H=70, W=60, title="Faces")

    p1,m1,s1,t1 = zip(*stats_d)
    p2,m2,s2,t2 = zip(*stats_f)

    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(p1,m1,linewidth=2,label="Digits")
    plt.plot(p2,m2,linewidth=2,label="Faces")
    plt.title("Test Accuracy vs. Training Size")
    plt.xlabel("Training (%)"); plt.ylabel("Accuracy")
    plt.legend(); plt.show()

    # Std Dev
    plt.figure(figsize=(8,5))
    plt.plot(p1,s1,linewidth=2,label="Digits")
    plt.plot(p2,s2,linewidth=2,label="Faces")
    plt.title("Std Dev of Accuracy")
    plt.xlabel("Training (%)"); plt.ylabel("Std Dev")
    plt.legend(); plt.show()

    # Time
    plt.figure(figsize=(8,5))
    plt.plot(p1,t1,linewidth=2,label="Digits")
    plt.plot(p2,t2,linewidth=2,label="Faces")
    plt.title("Training Time vs. Training Size")
    plt.xlabel("Training (%)"); plt.ylabel("Seconds")
    plt.legend(); plt.show()
