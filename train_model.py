import typer
import torch 
from torchvision import transforms, datasets
import os 
import time 
import copy
import models
from torch import nn 
from torch.optim import lr_scheduler



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
app = typer.Typer()


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

@app.command()
def train(data_dir,save_path,model,num_epochs, saved_state_dict = None):

    if model == 'baseline':
        model = models.SimpleConvModel()
    elif model == 'xception':
        model = models.Xception()
    elif model == 'effnet_b3':
        model = models.EfficientNet('b3')
    elif model == 'effnet_v2s':
        model = models.effnetv2_s()
    else:
        print("""
            Please input the model name:
            baseline
            xception
            effnet_b3
            effnet_v2s
        """)
    if saved_state_dict:
        model.load_state_dict(torch.load(saved_state_dict,map_location=torch.device('cpu')))
    
    model.to(device)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'valid']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                shuffle=True, num_workers=1)
                for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    print(dataset_sizes)
    class_names = image_datasets['train'].classes
    print(class_names)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    i=0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validationidation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evalidationuate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1).float()
    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = torch.sigmoid(outputs)
                    preds = outputs>0.5

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        i+=1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.data == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))
 
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)

