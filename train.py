import json, torch, torch.nn as nn
from torchvision import datasets, models, transforms as T
from torch.utils.data import DataLoader, random_split, Subset

EPOCHS = 12
norm = T.Normalize([.485, .456, .406], [.229, .224, .225])
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                      T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
val_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])

ds_train = datasets.ImageFolder("dataset", train_tf)
ds_val = datasets.ImageFolder("dataset", val_tf)
n = int(.8 * len(ds_train))
tr, va_split = random_split(ds_train, [n, len(ds_train) - n], generator=torch.Generator().manual_seed(42))
va = Subset(ds_val, va_split.indices)   # same split as evaluate.py, no augmentation
loader = lambda d, s: DataLoader(d, 32, shuffle=s, num_workers=0)

counts = torch.bincount(torch.tensor([ds_train.targets[i] for i in tr.indices]))
weights = counts.sum() / (len(counts) * counts)   # rarer class = bigger weight

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = models.mobilenet_v2(weights="DEFAULT")
for p in model.features.parameters():
    p.requires_grad = False
for p in model.features[14:].parameters():        # unfreeze last blocks
    p.requires_grad = True
model.classifier[1] = nn.Linear(1280, len(ds_train.classes))
model.to(dev)

opt = torch.optim.AdamW([
    {"params": model.features[14:].parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3}])
loss_fn = nn.CrossEntropyLoss(weight=weights.to(dev))

best = 0
for e in range(EPOCHS):
    model.train()
    for x, y in loader(tr, True):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
    model.eval(); ok = 0
    with torch.no_grad():
        for x, y in loader(va, False):
            ok += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
    acc = ok / len(va)
    print(f"epoch {e+1}/{EPOCHS}  val acc {acc:.3f}")
    if acc > best:                                 # keep only the best model
        best = acc
        torch.save(model.state_dict(), "backend/model.pt")

json.dump(ds_train.classes, open("backend/classes.json", "w"))
print(f"best val acc {best:.3f}")