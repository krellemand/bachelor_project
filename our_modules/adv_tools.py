import torch
import os

# These are the different loss funcitons used in the advesarial attacks. There reasoning behind the different losses is described in the Methodology chapter of our paper.

def log_msp_loss(y_hat, y):
    """Log Maximum Softmax Probablity Loss"""
    return torch.log(torch.amax(torch.exp(y_hat), dim=-1)/torch.sum(torch.exp(y_hat), dim=-1))


def norm_loss(y_hat, y, **norm_kwargs):
    """Used to create the 2 norm Loss"""
    return torch.linalg.norm(y_hat, **norm_kwargs)


def sum_exp_loss(y_hat, y, dim=-1):
    """Sum of Exponetials Loss"""
    return torch.sum(torch.exp(y_hat), dim=dim)


def max_loss(y_hat, y, dim=-1):
    """The max loss"""
    return torch.amax(y_hat, dim=dim)


def iterative_attack(model, xs, ys, loss_func, torch_optim, clip_range=(None, None), eps=None, return_step=False, max_iter=100, threshold=None, adv_type='fn', score_func= lambda logits: torch.amax(logits, dim=-1), **opt_kwargs):
    """
    Creates an Advesararial Example using a itterative method for the inputs "xs" that influses the model "model" prediction.
    The true labels "ys" are only used if the loss function is dependent on the labels of the images.

    loss_func:      The loss funciton that is maximized
    clip_range:     The range for pixel values. This is calulated based on the normilization used in preeprossesing. This is very dependent on the dataset.
    eps:            The range for how much a single pixel value can change
    return_step:    Weather the pertubation is returned seprately.
    max_iter:       Number of iterations
    threshold:      Weather the model should stop for a this given threshold.
    adv_type:       The sype of attack only used for thresholds.
    score_func:     The scoring funciton is also only used for if threshold is wanted with a given scoring function.
    """
    device = xs.device
    if clip_range != (None, None):
        clip_range = tuple([t.to(device) for t in clip_range])

    if len(xs.shape) == 3:
        x = xs
        y = ys
        x.requires_grad = True
        x_init = torch.clone(x).detach()
        optimizer = torch_optim([x], **opt_kwargs)
        i = 0

        while i < max_iter:
            optimizer.zero_grad()
            logits = model(x[None])
            loss = loss_func(logits, y)
            if threshold is not None:
                score = score_func(logits)
                if adv_type == 'fn':
                    to_stop = (score > threshold)
                elif adv_type == 'fp':
                    to_stop = (score < threshold)
                else:
                    raise NotImplementedError
                if to_stop:
                    break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if eps is not None:
                    x.copy_(torch.clip(x, min=x_init - eps*torch.ones_like(x_init), max=x_init + eps*torch.ones_like(x_init)))
                x.copy_(torch.clip(x, min=clip_range[0], max=clip_range[1]))
            x.requires_grad = True
            i += 1

        if return_step:
            step = x - x_init
            return x, step
        return x
    
    elif len(xs.shape) == 4:
        output = []
        steps = []
        for x, y in zip(xs,ys):
            x = torch.clone(x).detach()
            x.requires_grad = True
            x_init = torch.clone(x).detach()
            optimizer = torch_optim([x], **opt_kwargs)
            i = 0

            while i < max_iter:
                optimizer.zero_grad()
                logits = model(x[None])
                loss = loss_func(logits, y)

                if threshold is not None:
                    score = score_func(logits)
                    if adv_type == 'fn':
                        to_stop = (score > threshold)
                    elif adv_type == 'fp':
                        to_stop = (score < threshold)
                    else:
                        raise NotImplementedError
                    if to_stop:
                        break

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    if eps is not None:
                        x.copy_(torch.clip(x, min=x_init - eps*torch.ones_like(x_init), max=x_init + eps*torch.ones_like(x_init)))    
                    x.copy_(torch.clip(x, min=clip_range[0], max=clip_range[1]))

                x.requires_grad = True
                i += 1

            step = x - x_init
            output.append(x[None])
            steps.append(step[None])

        if return_step:
            return torch.cat(output), torch.cat(steps)
        return torch.cat(output)


# We are here creating iterative attacks for each type of attack. FP = False Posetive and FN = False Negative

def fp_osr_itat(model, x, eps=0.05, clip_range=(None, None), return_step=False, norm_ord=None, max_iter=25, torch_opt=torch.optim.Rprop, threshold=None, adv_type='fp', score_func= lambda logits: torch.amax(logits, dim=-1), **opt_kwargs):
    return iterative_attack(model, x, torch.zeros(len(x)), lambda yhat, y: norm_loss(yhat, y, ord=norm_ord, dim=-1), torch_opt, clip_range=clip_range, eps=eps, return_step=return_step, max_iter=max_iter, threshold=threshold, adv_type=adv_type, score_func=score_func, **opt_kwargs)  


def fn_osr_itat(model, x, eps=0.05, clip_range=(None, None), return_step=False, norm_ord=None, max_iter=25, torch_opt=torch.optim.Rprop, threshold=None, adv_type='fn', score_func= lambda logits: torch.amax(logits, dim=-1), **opt_kwargs):
    return iterative_attack(model, x, torch.zeros(len(x)), lambda yhat, y: -norm_loss(yhat, y, ord=norm_ord, dim=-1), torch_opt, clip_range=clip_range, eps=eps, return_step=return_step, max_iter=max_iter, threshold=threshold, adv_type=adv_type, score_func=score_func, **opt_kwargs)  


def fp_osr_itat_max(model, x, eps=0.05, clip_range=(None, None), return_step=False, max_iter=25, torch_opt=torch.optim.Rprop, threshold=None, adv_type='fp', score_func= lambda logits: torch.amax(logits, dim=-1), **opt_kwargs):
    return iterative_attack(model, x, torch.zeros(len(x)), lambda yhat, y: max_loss(yhat, y, dim=-1), torch_opt, clip_range=clip_range, eps=eps, return_step=return_step, max_iter=max_iter, threshold=threshold, adv_type=adv_type, score_func=score_func, **opt_kwargs)  


def fn_osr_itat_max(model, x, eps=0.05, clip_range=(None, None), return_step=False, max_iter=25, torch_opt=torch.optim.Rprop, threshold=None, adv_type='fn', score_func= lambda logits: torch.amax(logits, dim=-1), **opt_kwargs):
    return iterative_attack(model, x, torch.zeros(len(x)), lambda yhat, y: -max_loss(yhat, y, dim=-1), torch_opt, clip_range=clip_range, eps=eps, return_step=return_step, max_iter=max_iter, threshold=threshold, adv_type=adv_type, score_func=score_func, **opt_kwargs)


def fp_osr_itat_sum_exp(model, x, eps=0.05, clip_range=(None, None), return_step=False, max_iter=25, torch_opt=torch.optim.Rprop, threshold=None, adv_type='fp', score_func= lambda logits: torch.amax(logits, dim=-1), **opt_kwargs):
    return iterative_attack(model, x, torch.zeros(len(x)), lambda yhat, y: sum_exp_loss(yhat, y, dim=-1), torch_opt, clip_range=clip_range, eps=eps, return_step=return_step, max_iter=max_iter, threshold=threshold, adv_type=adv_type, score_func=score_func, **opt_kwargs) 



def fgsm(model, xs, ys, eps, loss_func, clip_range=(None, None), return_step=False, **loss_kwargs):
    """
    Creates an Advesararial Example with the fgsm for the inputs "xs" that influses the models "model" prediction.
    The true labels "ys" are only used if the loss function is dependent on the labels of the images.
    
    loss_func:      The loss funciton that is maximized
    clip_range:     The range for pixel values. This is calulated based on the normilization used in preeprossesing. This is very dependent on the dataset.
    eps:            The range for how much a single pixel value can change
    return_step:    Weather the pertubation is returned seprately.
    loss_kwargs:    Extra inputs for a loss function
    """
    device = xs.device
    if clip_range != (None, None):
        clip_range = tuple([t.to(device) for t in clip_range])
    if len(xs.shape) == 3:
        xs.requires_grad = True
        loss = loss_func(model(xs), ys, **loss_kwargs)
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
            loss = loss_func(model(x[None]), y, **loss_kwargs)
            loss.backward()
            step = eps*torch.sign(x.grad)
            steps.append(step[None])
            output.append((x + step)[None])

        if return_step:
            return torch.clip(torch.cat(output), clip_range[0], clip_range[1]), torch.cat(steps)
        return torch.clip(torch.cat(output), clip_range[0], clip_range[1])
    
    else:
        raise NotImplementedError


# We are here creating single step attacks using the fast gradient sign method for each type of attack. FP = False Posetive and FN = False Negative

def fp_osr_fgsm(model, x, eps=0.05, clip_range=(None, None), return_step=False, norm_ord=None):
    return fgsm(model, x, torch.zeros(len(x)), -eps, norm_loss, dim=-1, ord=norm_ord, 
                clip_range=clip_range, return_step=return_step)  


def fn_osr_fgsm(model, x, eps=0.05, clip_range=(None, None), return_step=False, norm_ord=None):
    return fgsm(model, x, torch.zeros(len(x)), eps, norm_loss, dim=-1, ord=norm_ord, 
                clip_range=clip_range, return_step=return_step)


def fp_osr_fgsm_max(model, x, eps=0.05, clip_range=(None, None), return_step=False, **throw_away_args):
    return fgsm(model, x, torch.zeros(len(x)), -eps, max_loss, clip_range=clip_range, return_step=return_step)


def fn_osr_fgsm_max(model, x, eps=0.05, clip_range=(None, None), return_step=False, **throw_away_args):
    return fgsm(model, x, torch.zeros(len(x)), eps, max_loss, clip_range=clip_range, return_step=return_step)


def fp_osr_fgsm_sum_exp(model, x, eps=0.05, clip_range=(None, None), return_step=False):
    return fgsm(model, x, torch.zeros(len(x)), -eps, sum_exp_loss, 
                clip_range=clip_range, return_step=return_step)


def fn_osr_fgsm_sum_exp(model, x, eps=0.05, clip_range=(None, None), return_step=False):
    return fgsm(model, x, torch.zeros(len(x)), eps, sum_exp_loss, 
                clip_range=clip_range, return_step=return_step)


def fn_osr_fgsm_log_msp(model, x, eps=0.05, clip_range=(None, None), return_step=False):
    return fgsm(model, x, torch.zeros(len(x)), eps, log_msp_loss, 
                clip_range=clip_range, return_step=return_step)

# Used to save the gradient norm for an input for a model and a function
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
            
            

