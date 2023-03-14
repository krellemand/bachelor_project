import torch

def fgsm(model, xs, ys, eps, loss_func):
    if len(xs.shape) == 3:
        xs.requires_grad = True
        loss = loss_func(model(xs), ys)
        loss.backward()
        return xs + eps*torch.sign(xs.grad)
    elif len(xs.shape) == 4:
        output = []
        for x, y in zip(xs,ys):
            x.requires_grad = True
            loss = loss_func(model(x[None]), y)
            loss.backward()
            output.append((x + eps*torch.sign(x.grad))[None])
        return torch.cat(output)
    else:
        raise NotImplementedError
    

def fp_osr_fgsm(model, x, eps=0.05):
    return fgsm(model, x, torch.zeros(len(x)), eps, lambda y_hat, y: torch.linalg.norm(y_hat, dim=-1, ord=torch.inf))

def fn_osr_fgsm(model, x, eps=0.05):
    return fgsm(model, x, torch.zeros(len(x)), -eps, lambda y_hat, y: torch.linalg.norm(y_hat, dim=-1, ord=None))