import torch
import os

def log_msp_loss(y_hat, y):
    return torch.log(torch.amax(torch.exp(y_hat), dim=-1)/torch.sum(torch.exp(y_hat), dim=-1))

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
    return fgsm(model, x, torch.zeros(len(x)), eps, log_msp_loss, 
                clip_range=clip_range, return_step=return_step)

def save_grad_norms(loss_func, model, dataloader, logdir, device, split_num, **norm_kwargs):
    model = model.to(device)
    grad_norms = []
    csr_targets = []
    uq_idxs = []

    for input_batch, target_batch, uq_idx in dataloader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        csr_targets += target_batch.tolist()
        uq_idxs += uq_idx.tolist()

        for x, y in zip(input_batch, target_batch):
            x.requires_grad = True
            loss = loss_func(model(x[None]), y)
            loss.backward()
            x_grad = x.grad
            grad_norm = torch.linalg.norm(torch.flatten(x_grad), **norm_kwargs)
            grad_norms.append(grad_norm)
    
    if logdir is not None:
        os.makedirs(logdir, exist_ok = True)
        torch.save(torch.tensor(grad_norms), logdir + "grad_norms_" + str(split_num) + ".pt")
        torch.save(torch.tensor(csr_targets), logdir + "csr_targets_" + str(split_num) + ".pt")
        torch.save(torch.tensor(uq_idxs), logdir + "index_" + str(split_num) + ".pt")
            
            

