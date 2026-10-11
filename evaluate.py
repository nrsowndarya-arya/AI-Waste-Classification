import json, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torchvision import datasets, models, transforms as T
from torch.utils.data import DataLoader, random_split

classes = json.load(open("backend/classes.json"))
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([.485, .456, .406], [.229, .224, .225])])
ds = datasets.ImageFolder("dataset", tf)
n = int(.8 * len(ds))
_, va = random_split(ds, [n, len(ds) - n], generator=torch.Generator().manual_seed(42))

model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(1280, len(classes))
model.load_state_dict(torch.load("backend/model.pt", map_location="cpu"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in DataLoader(va, 32):
        y_pred += model(x).argmax(1).tolist()
        y_true += y.tolist()

print(classification_report(y_true, y_pred, target_names=classes, digits=3))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(cmap="Greens", xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")