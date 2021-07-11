

import torch 
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import models 
import typer

testTransforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = typer.Typer()

@app.command()
def test(model, path):
    if model == 'baseline':
        model = models.SimpleConvModel()
        model.load_state_dict(torch.load('SimpleConv.pth',map_location=torch.device('cpu')))
    elif model == 'xception':
        model = models.Xception()
        model.load_state_dict(torch.load('Xception.pth',map_location=torch.device('cpu')))
    elif model == 'effnet_b3':
        model = models.EfficientNet('b3')
        model.load_state_dict(torch.load('EfficientNetB3.pth',map_location=torch.device('cpu')))
    elif model == 'effnet_v2s':
        model = models.effnetv2_s()
        model.load_state_dict(torch.load('efficientnet_v2_s.pth',map_location=torch.device('cpu')))
    else:
        print("""
            Please input the model name:
            baseline
            xception
            effnet_b3
            effnet_v2s
        """)
    model.to(device)
    test_dir = path
    testset = datasets.ImageFolder(test_dir,testTransforms)

    testLoader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=1)
              
    model.eval()

    y_true, y_pred = [], []
    for inputs, labels in testLoader:
        inputs = inputs.to(device)

        labels = labels.to(device)
        labels = labels.unsqueeze(1).float()

        y_pred.extend(model(inputs).sigmoid().flatten().tolist())
        y_true.extend(labels.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)<0.5
    y_true = (y_true-1)*(-1)
    pre = precision_score(y_true,y_pred )
    rec = recall_score(y_true,y_pred )
    acc = accuracy_score(y_true, y_pred )

    print('Tested on: {}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}% '.format(path, acc*100,pre*100,rec*100))


if __name__ == "__main__":
    app()