import torch 
import torch.optim as optim
from torch import linalg as LA


def fgs(model, input, target, eps=0.05):

    model.eval()
    model.zero_grad()
    input_var = torch.autograd.Variable(input, requires_grad=True)
    output = model(input_var).sigmoid()

    criterion = torch.nn.BCELoss()
    loss = criterion(output, target)
    loss.backward()

    grad_sign = input_var.grad.sign()


    return input_var + eps * grad_sign 

def clipping(adv, og, eps):
    up = torch.clamp(og+eps,max=2.59)
    low = torch.clamp(og-eps, min=-2.12)

    tmp = torch.where(adv > og + eps, up, adv)
    tmp = torch.where(adv < og - eps, low, adv)
    return tmp.detach()

def basic_iterative_attack(model, inputs, labels, eps = 0.05, alpha=0.05, iters=10) :

    if iters == 0 :
        # The paper said min(eps + 4, 1.25*eps) is used as iterations
        iters = int(min(eps*255 + 4, 1.25*eps*255))    

    criterion = torch.nn.BCELoss()
    input_var = torch.autograd.Variable(inputs, requires_grad=True)

    for i in range(iters) :    
        input_var.requires_grad=True

        outputs = model(input_var).sigmoid()
        #if all predicts are wrong: stop
        inverse_pred = outputs < 0.5
        if torch.all(torch.eq(inverse_pred,labels)):
            return input_var

    #calculate grad 
    model.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
  
    #update input
    input_var = input_var + alpha*input_var.grad.sign()
    input_var = clipping(input_var, inputs, eps)

    return input_var


def white_box_attack(model, inputs, labels, eps = 0.05, alpha=0.05, iters=10) :

    criterion = lambda input,target : (input*(target*2-1)).mean()
    input_var = inputs

    for i in range(iters) :    
        input_var.requires_grad=True
        outputs = model(input_var)
        #if all predicts are wrong: stop
        inverse_pred = outputs < 0.5
    
        if torch.all(torch.eq(inverse_pred,labels)):
            return input_var
    
        #calculate grad 
        model.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        
        #update input
        input_var = input_var - alpha*input_var.grad.sign()
        input_var = clipping(input_var, inputs, eps)

    return input_var


def universal_attack(model, inputs, labels, eps = 0.12,alpha=0.01, iters = 100):
    criterion = torch.nn.BCELoss()
    model.eval()  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p = torch.randn(inputs.shape).to(device)
    p.requires_grad = True
    optimizer = optim.Adam([p], lr=0.001)

    for i in range(iters):
        delta = torch.tanh(p)*eps
        outputs = model(inputs+delta).sigmoid()

        inverse_pred = outputs < 0.5
        if torch.all(torch.eq(inverse_pred, labels)):
            return inputs + delta

        model.zero_grad()

        loss = criterion(outputs , -(labels-1)) + alpha*torch.dist(torch.zeros(delta.shape).to(device),delta)
        loss.backward()
        optimizer.step()

    return inputs+delta
  

# #ref https://github.com/Jianbo-Lab/HSJA
# def project(og, perturbed, alpha, constraint):

# 	if constraint == 'l2':
# 		return (1-alpha) * og + alpha * perturbed
# 	elif constraint == 'linf':
# 		out_images = clipping(perturbed, og, alpha)
# 		return out_images


# def bin_search(og, perturbed, model,constraint, theta, label):

#     dists_post_update = torch.dist(og,perturbed) if constraint == 'l2' else (og,perturbed).abs().max()

#     high = 1 if constraint == 'l2' else dists_post_update
#     thresh = theta if constraint == 'l2' else min(dists_post_update*theta,theta)

#     low = 0

#     while ((high-low)/thresh) >1:
#         mid = (high + low)/2.0
#         mid_img = project(og,perturbed,mid,constraint)

#         decision = (model(mid_img)>0.5)==label

def black_box_NES(model, inputs, labels,eps=0.05,alpha = 0.05, var = 0.01, n=25, iters=100 ):
    model.eval()
    criterion = torch.nn.BCELoss()
    input_var = inputs
    for i in range(iters):
        est = torch.zeros(input_var.shape).to(inputs.device)

        outputs = model(input_var)
        #if all predicts are wrong: stop
        inverse_pred = outputs.sigmoid() < 0.5

        if torch.all(torch.eq(inverse_pred,labels)):

            return inputs

        #estimate the gradient
        for j in range(n):
            r = torch.randn(input_var.shape).to(inputs.device)
            out1 = criterion(model(input_var + r*var).sigmoid(),labels).to(inputs.device)
            out2 = criterion(model(input_var - r*var).sigmoid(),labels).to(inputs.device)
            est += (r*out1).detach()
            est -= (r*out2).detach()
        #update input
        input_var = input_var + alpha*est.sign()
        input_var = clipping(input_var, inputs, eps)

    return input_var

def deepfool(model,inputs,labels,eps=0.05, iters = 10):
    input_var = inputs

    for i in range(iters):
        input_var.requires_grad = True

        outputs = model(input_var)
        inverse_pred = outputs.sigmoid() < 0.5
        if torch.all(torch.eq(inverse_pred,labels)):
            return input_var
        outputs.backward()
        input_var = input_var - 1.02*input_var.grad*outputs/LA.norm(input_var.grad)**2
        input_var = clipping(input_var,inputs,eps)

    return input_var

# https://github.com/cg563/simple-blackbox-attack/blob/master/simba.py
def simba(model, inputs, labels, eps= 0.2 , iters = 10000):
    dims = inputs.view(1,-1).size(1)
    perm = torch.randperm(dims).to(inputs.device)
    last_prob = model(inputs).sigmoid()
    last_prob = last_prob if labels == 1 else 1 - last_prob
    for i in range(iters):
        diff = torch.zeros(dims).to(inputs.device)
        diff[perm[i]] = eps
        left_prob = model((inputs-diff.view(inputs.size())).clamp(-2.12,2.59)).sigmoid()
        left_prob = left_prob if labels == 1 else 1 - left_prob
    
        if left_prob < last_prob:
            inputs = (inputs-diff.view(inputs.size())).clamp(-2.12,2.59)
            last_prob = left_prob
            
        else:
            right_prob = model((inputs+diff.view(inputs.size())).clamp(-2.12,2.59)).sigmoid()
            right_prob = right_prob if labels == 1 else 1 - right_prob
            if right_prob < last_prob:
                inputs = (inputs+diff.view(inputs.size())).clamp(-2.12,2.59)
                last_prob = right_prob
            
        if i % 1000 == 0 :
            outputs = model(inputs).sigmoid()
            #if all predicts are wrong: stop
            inverse_pred = outputs < 0.5
            if torch.all(torch.eq(inverse_pred,labels)):
                return inputs
    return inputs 