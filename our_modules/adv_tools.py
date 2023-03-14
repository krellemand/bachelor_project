import torch

def fgsm(model, x, y, eps, loss_func):
    x.requires_grad = True
    loss = loss_func(model(x), y)
    loss.backward()
    return x + eps*torch.sign(x.grad)

def fp_osr_fgsm(model, x, eps=0.05):
    return fgsm(model, x, None, eps, lambda y_hat, y: torch.linalg.norm(y_hat, dim=-1, ord=torch.inf))

def fn_osr_fgsm(model, x, eps=0.05):
    return fgsm(model, x, None, -eps, lambda y_hat, y: torch.linalg.norm(y_hat, dim=-1, ord=torch.inf))