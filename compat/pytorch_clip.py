import jittor as jt


def clip_grad_norm_pytorch(optimizer, max_norm, norm_type=2):
    gradients = []
    for param_group in optimizer.param_groups:
        for parameter, gradient in zip(
            param_group["params"], param_group["grads"]
        ):
            if not parameter.is_stop_grad():
                gradients.append(gradient)
    if not gradients:
        return jt.array(0.0)
    parameter_norms = jt.stack(
        [jt.norm(gradient.flatten(), norm_type) for gradient in gradients]
    )
    total_norm = jt.norm(parameter_norms.flatten(), norm_type)
    clip_coefficient = jt.minimum(max_norm / (total_norm + 1e-6), 1.0)
    for gradient in gradients:
        gradient.update(gradient * clip_coefficient)
    return total_norm
