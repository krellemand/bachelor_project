import torch

def fgsm(model, xs, ys, eps, loss_func, clip_range=(None, None)):
    if len(xs.shape) == 3:
        xs.requires_grad = True
        loss = loss_func(model(xs), ys)
        loss.backward()
        return torch.clip(xs + eps*torch.sign(xs.grad), clip_range[0], clip_range[1])
    elif len(xs.shape) == 4:
        output = []
        for x, y in zip(xs,ys):
            x.requires_grad = True
            loss = loss_func(model(x[None]), y)
            loss.backward()
            output.append((x + eps*torch.sign(x.grad))[None])
        return torch.clip(torch.cat(output), clip_range[0], clip_range[1])
    else:
        raise NotImplementedError
    

def fp_osr_fgsm(model, x, eps=0.05, clip_range=(None, None)):
    return fgsm(model, x, torch.zeros(len(x)), -eps, lambda y_hat, y: torch.linalg.norm(y_hat, dim=-1, ord=None), clip_range=clip_range)

def fn_osr_fgsm(model, x, eps=0.05, clip_range=(None, None)):
    return fgsm(model, x, torch.zeros(len(x)), eps, lambda y_hat, y: torch.linalg.norm(y_hat, dim=-1, ord=torch.inf), clip_range=clip_range)