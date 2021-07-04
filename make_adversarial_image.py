from torchvision.transforms.transforms import PILToTensor, ToPILImage
from dataset import SingleDirDataset, fakeNrealDataset
import attacks 
import models 
import typer
from torchvision import transforms
import torch 
from pytorch_msssim import ssim
import numpy as np
from torchvision.utils import save_image
import os

app = typer.Typer()

@app.command()
def single_dir(path, label,out_path,att,model = 'baseline'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if label == 'fake': label = 0
    elif label == 'real': label = 1
    label = torch.FloatTensor([[label]])
    label.unsqueeze(1).to(device)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = SingleDirDataset(path, data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True, num_workers=1)
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
    ssim_mean = 0
    i = 0
    for input in dataloader:
        input = input.to(device)
        raw = input
        if att == 'fgs':
            input = attacks.fgs(model,input,label)
        elif att == 'bce_iter':
            input = attacks.basic_iterative_attack(model,input,label)
        elif att == 'bce_iter_tran':
            input = attacks.basic_iterative_attack(model,input,label,trans=True)
        elif att == 'lin_iter':
            input = attacks.white_box_attack(model,input,label)        
        elif att == 'lin_iter_tran':
            input = attacks.white_box_attack(model,input,label,trans=True)
        elif att == 'deepfool':
            input = attacks.deepfool(model,input,label)
        elif att == 'nes':
            input = attacks.black_box_NES(model,input,label)
        elif att == 'nes_tran':
            input = attacks.black_box_NES(model,input,label,trans=True)
        elif att == 'simba':
            input = attacks.simba(model,input,label)
        else:
            print("""please input one of these the attack method:
            fgs
            bce_iter
            bce_iter_tran
            lin_iter
            lin_iter_tran
            deepfool
            nes
            nes_tran
            simba
            """)
            return
        input = img_denorm(input).unsqueeze(0)
        ssim_val = ssim( img_denorm(raw).unsqueeze(0), input, data_range=1, size_average=True)
        save_image(input, out_path+'/{}.png'.format(i) )
        i += 1
        ssim_mean += ssim_val.item()


    ssim_mean = ssim_mean/i
    print(f'{att} SSIM: {ssim_mean}')
        

@app.command()
def fakeNrealdir(path,out_path,att,model = 'baseline'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(out_path):
        os.makedirs(os.path.join(out_path,'fake'))
        os.makedirs(os.path.join(out_path,'real'))

    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = fakeNrealDataset(path, data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True, num_workers=1)
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
    ssim_mean = 0
    i = 0
    fcount = 0
    rcount = 0
    for input, label in dataloader:
        input = input.to(device)
        label = label.unsqueeze(1).float().to(device)
        raw = input
        if att == 'fgs':
            input = attacks.fgs(model,input,label)
        elif att == 'bce_iter':
            input = attacks.basic_iterative_attack(model,input,label)
        elif att == 'bce_iter_tran':
            input = attacks.basic_iterative_attack(model,input,label,trans=True)
        elif att == 'lin_iter':
            input = attacks.white_box_attack(model,input,label)
        elif att == 'lin_iter_tran':
            input = attacks.white_box_attack(model,input,label,trans=True)
        elif att == 'deepfool':
            input = attacks.deepfool(model,input,label)
        elif att == 'nes':
            input = attacks.black_box_NES(model,input,label)
        elif att == 'nes_tran':
            input = attacks.black_box_NES(model,input,label,trans=True)
        elif att == 'simba':
            input = attacks.simba(model,input,label)
        else:
            print("""please input one of these the attack method:
            fgs
            bce_iter
            bce_iter_tran
            lin_iter
            lin_iter_tran
            deepfool
            nes
            nes_tran
            simba
            """)
        input = img_denorm(input).unsqueeze(0)
        ssim_val = ssim( img_denorm(raw).unsqueeze(0), input, data_range=1, size_average=True)

        i += 1
        ssim_mean += ssim_val.item()

        if label.item() == 0:
            save_image(input, out_path+'/fake/{}.png'.format(fcount))
            fcount += 1
        else: 
            save_image(input, out_path+'/real/{}.png'.format(rcount))
            rcount += 1 
    ssim_mean = ssim_mean/i
    print(f'{att} SSIM: {ssim_mean}')


def img_denorm(img):
    #for ImageNet the mean and std are:
    mean = np.asarray([ 0.485, 0.456, 0.406 ])
    std = np.asarray([ 0.229, 0.224, 0.225 ])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0)
    res = denormalize(res)

    #Image needs to be clipped since the denormalize function will map some
    #values below 0 and above 1
    res = torch.clamp(res, 0, 1)
    return res


if __name__ == "__main__":
    app()