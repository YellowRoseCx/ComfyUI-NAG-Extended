import math
import torch


def nag(z_positive, z_negative, scale, tau, alpha):
    m = min(z_positive.shape[0], z_negative.shape[0])
    if m == 0:
        return z_negative
        
    z_pos_slice = z_positive[-m:]
    z_neg_slice = z_negative[-m:]
    
    # 1. Allocate the guidance tensor ONCE
    z_guidance = torch.empty_like(z_pos_slice)
    z_guidance.copy_(z_pos_slice)
    
    # 2. In-place math for extrapolation: z_guidance = z_pos * scale - z_neg * (scale - 1)
    z_guidance.mul_(scale)
    z_guidance.sub_(z_neg_slice, alpha=(scale - 1))
    
    eps = 1e-6
    
    # 3. Calculate norms WITHOUT .expand_as() 
    # We leave these as smaller broadcastable shapes, saving massive amounts of VRAM
    norm_positive = torch.norm(z_pos_slice, p=1, dim=-1, keepdim=True).clamp_min_(eps)
    norm_guidance = torch.norm(z_guidance, p=1, dim=-1, keepdim=True).clamp_min_(eps)
    
    # 4. In-place calculation for s
    s = norm_guidance.div_(norm_positive)
    
    # 5. In-place clamping logic
    tau_tensor = torch.full((1,), tau, dtype=s.dtype, device=s.device)
    clamp_factor = torch.minimum(s, tau_tensor).div_(s)
    
    # 6. Apply the clamp to the guidance tensor in-place
    z_guidance.mul_(clamp_factor)
    
    # 7. Final blend in-place: z_guidance * alpha + z_pos * (1 - alpha)
    z_guidance.mul_(alpha)
    z_guidance.add_(z_pos_slice, alpha=(1 - alpha))
    
    return z_guidance


def cat_context(context, nag_negative_context, trim_context=False, dim=1):
    assert dim in [1, 2]
    nag_negative_context = nag_negative_context.to(context)

    context_len = context.shape[dim]
    nag_neg_context_len = nag_negative_context.shape[dim]

    if context_len < nag_neg_context_len:
        if dim == 1:
            context = context.expand(-1, nag_neg_context_len, -1) if context_len == 1 else context.repeat(1, math.ceil(nag_neg_context_len / context_len), 1)
            if trim_context:
                context = context[:, -nag_neg_context_len:]
        else:
            context = context.expand(-1, -1, nag_neg_context_len, -1) if context_len == 1 else context.repeat(1, 1, math.ceil(nag_neg_context_len / context_len), 1)
            if trim_context:
                context = context[:, :, -nag_neg_context_len:]

        context_len = context.shape[dim]

    if dim == 1:
        nag_negative_context = nag_negative_context.expand(-1, context_len, -1) if nag_neg_context_len == 1 else nag_negative_context.repeat(1, math.ceil(context_len / nag_neg_context_len), 1)
        nag_negative_context = nag_negative_context[:, -context_len:]
    else:
        nag_negative_context = nag_negative_context.expand(-1, -1, context_len, -1) if nag_neg_context_len == 1 else nag_negative_context.repeat(1, 1, math.ceil(context_len / nag_neg_context_len), 1)
        nag_negative_context = nag_negative_context[:, :, -context_len:]


    return torch.cat([context, nag_negative_context], dim=0)


def check_nag_activation(transformer_options, nag_sigma_start, nag_sigma_end):
    """
    Check if NAG should be activated based on current sigma value.
    NAG is active when: nag_sigma_end <= current_sigma <= nag_sigma_start
    """
    sigmas = transformer_options.get("sigmas")
    if sigmas is None:
        return False
    
    # Check if all sigmas in the current batch are within the NAG activation range
    apply_nag = torch.all((sigmas >= nag_sigma_end) & (sigmas <= nag_sigma_start))
    positive_batch = 0 in transformer_options["cond_or_uncond"]
    return apply_nag and positive_batch


def get_closure_vars(func):
    if func.__closure__ is None:
        return {}
    return {
        var: cell.cell_contents
        for var, cell in zip(func.__code__.co_freevars, func.__closure__)
    }


def is_from_wavespeed(func):
    closure = get_closure_vars(func)
    return "residual_diff_threshold" in closure \
        and "validate_can_use_cache_function" in closure


class NAGSwitch:
    def __init__(
        self,
        model: torch.nn.Module,
        nag_negative_cond,
        nag_scale, nag_tau, nag_alpha, nag_sigma_start, nag_sigma_end,
    ):
        self.model = model
        self.nag_negative_cond = nag_negative_cond
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.nag_sigma_start = nag_sigma_start
        self.nag_sigma_end = nag_sigma_end
        self.origin_forward = model.forward

    def set_nag(self):
        pass

    def set_origin(self):
        self.model.forward = self.origin_forward


# https://github.com/welltop-cn/ComfyUI-TeaCache/blob/4bca908bf53b029ea5739cb69ef2a9e6c06e6752/nodes.py
def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result
