import math

import jittor as jt


class PyTorchAdam(jt.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=0,
    ):
        super().__init__(params, lr)
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay
        for param_group in self.param_groups:
            param_group["exp_avg"] = [
                jt.zeros(parameter.shape, parameter.dtype).stop_grad()
                for parameter in param_group["params"]
            ]
            param_group["exp_avg_sq"] = [
                jt.zeros(parameter.shape, parameter.dtype).stop_grad()
                for parameter in param_group["params"]
            ]

    def add_param_group(self, group):
        group["exp_avg"] = [
            jt.zeros(parameter.shape, parameter.dtype).stop_grad()
            for parameter in group["params"]
        ]
        group["exp_avg_sq"] = [
            jt.zeros(parameter.shape, parameter.dtype).stop_grad()
            for parameter in group["params"]
        ]
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        step = float(self.n_step)
        jt.flags.node_order = 1
        for param_group in self.param_groups:
            learning_rate = param_group.get("lr", self.lr)
            epsilon = param_group.get("eps", self.eps)
            weight_decay = param_group.get("weight_decay", self.weight_decay)
            beta1, beta2 = param_group.get("betas", self.betas)
            bias_correction1 = 1 - beta1**step
            bias_correction2_sqrt = math.sqrt(1 - beta2**step)
            for parameter, gradient, exp_avg, exp_avg_sq in zip(
                param_group["params"],
                param_group["grads"],
                param_group["exp_avg"],
                param_group["exp_avg_sq"],
            ):
                if parameter.is_stop_grad():
                    continue
                gradient = gradient + parameter * weight_decay
                exp_avg.update(beta1 * exp_avg + (1 - beta1) * gradient)
                exp_avg_sq.update(
                    beta2 * exp_avg_sq + (1 - beta2) * gradient * gradient
                )
                denominator = jt.sqrt(exp_avg_sq) / bias_correction2_sqrt + epsilon
                parameter.update(
                    parameter
                    - learning_rate / bias_correction1 * exp_avg / denominator
                )
        self.post_step()
