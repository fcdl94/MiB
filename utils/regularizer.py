import torch
from copy import deepcopy

EPS = 1e-8


def get_regularizer(model, model_old, device, opts, old_state):
    resume = False
    name = opts.regularizer
    if old_state is not None:
        if name != old_state['name']:
            print(f"Warning: the regularizer you passed {name}"
                  f" is different from the state one {old_state['name']}")
        resume = True

    if name is None:
        return None
    elif name == 'ewc':
        fisher = old_state["fisher"] if resume else None
        return EWC(model, model_old, device, fisher=fisher,
                   alpha=opts.reg_alpha,
                   normalize= not opts.reg_no_normalize)
    elif name == 'pi':
        score = old_state["score"] if resume else None
        return PI(model, model_old, device, score=score,
                  normalize=not opts.reg_no_normalize)
    elif name == 'rw':
        score = old_state["score"] if resume else None
        fisher = old_state["fisher"] if resume else None
        return RW(model, model_old, device, score=score, fisher=fisher,
                  alpha=opts.reg_alpha, iterations=opts.reg_iterations,
                  normalize=not opts.reg_no_normalize)
    else:
        raise NotImplementedError


def normalize_fn(mat):
    return (mat - mat.min()) / (mat.max() - mat.min() + EPS)


class Regularizer:
    def update(self):
        """ Stub method """
        raise NotImplementedError

    def penalty(self):
        """ Stub method """
        raise NotImplementedError

    def state_dict(self):
        """ Stub method """
        raise NotImplementedError

    def load_state_dict(self, state):
        """ Stub method """
        raise NotImplementedError


class EWC(Regularizer):
    # note: by taking in consideration the torch.distributed package and that the update is only computed by the rank 0,
    #       we can save memory in other ranks. Actually it's not useful because I use GPU with the same memory.
    def __init__(self, model, model_old, device, fisher=None, alpha=0.9, normalize=True):

        self.model = model
        self.device = device
        self.alpha = alpha
        self.normalize = normalize

        # store old model for penalty step
        if model_old is not None:
            self.model_old = model_old
            self.model_old_dict = self.model_old.state_dict()
            self.penalize = True
        else:
            self.penalize = False

        # make the fisher matrix for the estimate of parameter importance
        # store the old fisher matrix (if exist) for penalize step
        if fisher is not None:  # initialize the old Fisher Matrix
            self.fisher_old = fisher
            self.fisher = {}
            for key, par in self.fisher_old.items():
                self.fisher_old[key].requires_grad = False
                self.fisher_old[key] = normalize_fn(par) if normalize else par
                self.fisher_old[key] = self.fisher_old[key].to(device)
                self.fisher[key] = torch.clone(par).to(device)
        else:  # initialize a new Fisher Matrix and don't penalize, we miss an information
            self.fisher_old = None
            self.penalize = False
            self.fisher = {}

        for n, p in self.model.named_parameters():  # update fisher with new keys (due to incremental classes)
            if p.requires_grad and n not in self.fisher:
                self.fisher[n] = torch.ones_like(p, device=device, requires_grad=False)

    def update(self):
        # suppose model have already grad computed, so we can directly update the fisher by getting model.parameters
        for n, p in self.model.named_parameters():
            self.fisher[n] = (self.alpha * (p.grad ** 2)) + ((1 - self.alpha) * self.fisher[n])

    def penalty(self):
        if not self.penalize:
            return 0.
        else:
            loss = 0.
            for n, p in self.model.named_parameters():
                if n in self.model_old_dict and p.requires_grad:
                    loss += (self.fisher_old[n] * (p - self.model_old_dict[n]) ** 2).sum()
            return loss

    def get(self):
        return self.fisher  # return the new Fisher matrix

    def state_dict(self):
        state = {"name": "ewc", "fisher": self.fisher, "alpha": self.alpha,}
        return state

    def load_state_dict(self, state):
        assert state['name'] == 'ewc', f"Error, you are trying to restore {state['name']} into ewc"
        self.fisher = state["fisher"]
        for k,p in self.fisher.items():
            self.fisher[k] = p.to(self.device)
        self.alpha = state["alpha"]


class PI(Regularizer):  # Path integral
    # note: by taking in consideration the torch.distributed package and that the update is only computed by the rank 0,
    #       we can save memory in other ranks. Actually it's not useful because I use memory with the same size.

    def __init__(self, model, model_old, device, score, normalize=False):

        self.model = model

        self.device = device
        self.normalize = normalize
        self.penalize = True
        self.starting_new = {}

        if model_old is not None:
            self.model_old_dict = model_old.state_dict()
            for k, p in model.named_parameters():
                if k not in self.model_old_dict:
                    self.starting_new[k] = p.clone().detach().cpu()
                    new_p = torch.clone(p).detach().to(device)
                    self.model_old_dict[k] = new_p
        else:
            self.model_old_dict = deepcopy(model.state_dict())
            self.penalize = False

        if score is not None:
            self.score = score  # to compute the penalty term
            self.score_actual = {}
            for n, p in score.items():
                p.requires_grad = False
                self.score_actual[n] = normalize_fn(p.to(device)) if normalize else p.to(device)
        else:  # initialize a new Fisher Matrix
            self.score = None
            self.penalize = False

        self.delta = {n: torch.zeros_like(p, device=device, requires_grad=False)
                      for n, p in self.model.named_parameters()}
        self.model_temp = None  # to be updated at the first iteration

    def update(self):
        # to be called every t iteration and at the starting iteration with grad already computed
        if self.model_temp is not None:
            # update the score
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    delta = p.grad.detach() * (self.model_temp[n].to(self.device) - p.detach())
                    self.delta[n] += delta  # approximation of path integral
        # update model temp
        self.model_temp = {k: torch.clone(p).detach().cpu()
                           for k, p in self.model.named_parameters() if p.grad is not None}

    def penalty(self):
        loss = 0
        if not self.penalize:
            return 0.
        for n, p in self.model.named_parameters():
            if n in self.score_actual and p.requires_grad:
                loss += (self.score_actual[n] * (p - self.model_old_dict[n]).pow(2)).sum()
        return loss

    def get(self):
        score = {}
        EPS = 1e-20
        for n, p in self.model.named_parameters():
            score[n] = self.delta[n] / ((p.detach() - self.model_old_dict[n]).pow(2) + EPS)
            score[n] = torch.where(score[n] > 0, score[n], torch.tensor(0.).to(self.device))
            if self.score is not None and n in self.score:
                score[n] = self.score[n].to(self.device) + score[n]  # the importance is averaged
        return score  # return the score matrix

    def state_dict(self):
        state = {"name": "pi", "score": self.get(), "delta": self.delta,
                 "starting_model": self.starting_new}
        return state

    def load_state_dict(self, state):
        assert state['name'] == 'pi', f"Error, you are trying to restore {state['name']} into pi"
        self.delta = state['delta']
        for k, v in self.delta.items():
            self.delta[k] = v.to(self.device)
        for k, p in state['starting_model'].items():
            self.model_old_dict[k] = p.to(self.device)


class RW(Regularizer):
    # note: by taking in consideration the torch.distributed package and that the update is only computed by the rank 0,
    #       we can save memory in other ranks. Actually it's not useful because I use memory with the same size.
    def __init__(self, model, model_old, device, score, fisher,
                 alpha=0.9, iterations=10, normalize=True):

        self.model = model
        self.device = device

        self.alpha = alpha
        self.penalize = True
        self.normalize = normalize
        self.iterations = iterations

        self.count = 0

        if model_old is not None:
            self.model_old_dict = model_old.state_dict()
        else:
            self.model_old_dict = deepcopy(model.state_dict())
            self.penalize = False

        if fisher is not None and score is not None:  # initialize the old Fisher Matrix
            self.fisher = {}  # to compute online the fisher matrix
            self.score_plus_fisher = {}
            for key, par in fisher.items():
                par.requires_grad = False
                self.score_plus_fisher[key] = normalize_fn(par.to(device)) if normalize else par.to(device)
                self.fisher[key] = torch.clone(par).to(device)

            self.score_old = {}  # to compute the next score matrix -> not in GPU memory until used
            for n, p in score.items():
                p.requires_grad = False
                self.score_old[n] = p
                self.score_plus_fisher[n] += normalize_fn(p.to(device)) if normalize else p.to(device)
                if torch.isnan(self.score_plus_fisher[n].mean()) or torch.isinf(self.score_plus_fisher[n].mean()):
                    print("Some error here")

        else:
            self.penalize = False
            self.score_old = None
            self.fisher = {}

        self.score = {n: torch.zeros_like(p, device=device, requires_grad=False)  # to compute the new score matrix
                      for n, p in self.model.named_parameters() if p.requires_grad}

        for n, p in self.model.named_parameters():  # update fisher with new keys (due to incremental classes)
            if p.requires_grad and n not in self.fisher:
                self.fisher[n] = torch.ones_like(p, device=device, requires_grad=False)

        self.model_temp = None  # to be updated at the first iteration

    def update(self):
        # to be called every t iteration and at the starting iteration with grad already computed
        if self.count % self.iterations == 0:
            if self.model_temp is not None:
                # update the score
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        delta = p.grad.detach() * (self.model_temp[n].to(self.device) - p.detach())
                        den = 0.5 * self.fisher[n] * (p.detach() - self.model_temp[n].to(self.device)).pow(2) + EPS
                        self.score[n] += (delta / den)
            # update model temp
            self.model_temp = {k: torch.clone(p).detach().cpu()
                               for k, p in self.model.named_parameters() if p.grad is not None}
        self.count += 1

        for n, p in self.model.named_parameters():
            if p.grad is not None:
                # assert not (torch.isnan(p.grad.sum()) or torch.isinf(p.grad.sum())), "Here is a gradient that is nan"
                self.fisher[n] = (self.alpha * p.grad.detach().pow(2)) + ((1 - self.alpha) * self.fisher[n])

    def get_score(self):
        score = {}
        for n, p in self.score.items():
            score[n] = torch.where(p >= 0, p, torch.tensor(0.).to(self.device))
            if self.score_old is not None and n in self.score_old:
                score[n] = 0.5 * (score[n] + self.score_old[n].to(self.device))  # the importance is averaged
        return score  # return the score matrix

    def penalty(self):
        loss = 0
        if not self.penalize:
            return 0.
        for n, p in self.model.named_parameters():
            if n in self.model_old_dict and p.requires_grad:
                x = ((self.score_plus_fisher[n]) * (p - self.model_old_dict[n]).pow(2)).sum()
                loss += x
        return loss

    def state_dict(self):
        state = {"name": "rw", "score": self.get_score(), "fisher": self.fisher,
                 "iteration": self.iterations, "alpha": self.alpha}
        return state

    def load_state_dict(self, state):
        assert state['name'] == 'rw', f"Error, you are trying to restore {state['name']} into rw"
        self.iterations = state["iteration"]
        self.alpha = state['alpha']
        self.fisher = state['fisher']
        for k, p in self.fisher.items():
            self.fisher[k] = p.to(self.device)
        self.score = state['score']
        for k, p in self.score.items():
            self.score[k] = p.to(self.device)
