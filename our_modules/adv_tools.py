import torch

def fgsm(model, xs, ys, eps, loss_func, clip_range=(None, None), return_step=False):
    if len(xs.shape) == 3:
        xs.requires_grad = True
        loss = loss_func(model(xs), ys)
        loss.backward()
        step = eps*torch.sign(xs.grad)
        if return_step:
            return torch.clip(xs + step, clip_range[0], clip_range[1]), step
        return torch.clip(xs + step, clip_range[0], clip_range[1])
    elif len(xs.shape) == 4:
        output = []
        steps = []
        for x, y in zip(xs,ys):
            x.requires_grad = True
            loss = loss_func(model(x[None]), y)
            loss.backward()
            step = eps*torch.sign(x.grad)
            steps.append(step[None])
            output.append((x + step)[None])
        if return_step:
            return torch.clip(torch.cat(output), clip_range[0], clip_range[1]), torch.cat(steps)
        return torch.clip(torch.cat(output), clip_range[0], clip_range[1])
    else:
        raise NotImplementedError
    
def fp_osr_fgsm(model, x, eps=0.05, clip_range=(None, None), return_step=False, norm_ord=None):
    return fgsm(model, x, torch.zeros(len(x)), -eps, lambda y_hat, y: torch.linalg.norm(y_hat, dim=-1, ord=norm_ord), 
                clip_range=clip_range, return_step=return_step)

def fn_osr_fgsm(model, x, eps=0.05, clip_range=(None, None), return_step=False, norm_ord=torch.inf):
    return fgsm(model, x, torch.zeros(len(x)), eps, lambda y_hat, y: torch.linalg.norm(y_hat, dim=-1, ord=norm_ord), 
                clip_range=clip_range, return_step=return_step)

def fp_osr_fgsm_sum_exp(model, x, eps=0.05, clip_range=(None, None), return_step=False):
    return fgsm(model, x, torch.zeros(len(x)), -eps, lambda y_hat, y: torch.sum(torch.exp(y_hat), dim=-1), 
                clip_range=clip_range, return_step=return_step)

def fn_osr_fgsm_log_msp(model, x, eps=0.05, clip_range=(None, None), return_step=False):
    loss = lambda y_hat, y: torch.log(torch.amax(torch.exp(y_hat), dim=-1)/torch.sum(torch.exp(y_hat), dim=-1))
    return fgsm(model, x, torch.zeros(len(x)), eps, loss, 
                clip_range=clip_range, return_step=return_step)

def save_grad_norms(loss_func, model, dataloader, logdir, device):
    model = model.to(device)
    
    for input_batch, target_batch, uq_idx in dataloader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        grad_norms = []

        for x, y in zip(input_batch, target_batch):
            x.requires_grad = True
            loss = loss_func(model(x[None]), y)
            loss.backward()
            

