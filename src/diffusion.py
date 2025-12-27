import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

class DDPM:
    def __init__(self, T: int = 1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.T = T
        self.device = device

        betas = linear_beta_schedule(T, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        x_t = sqrt(a_bar_t)*x0 + sqrt(1-a_bar_t)*noise
        """
        b = x0.shape[0]
        s1 = self.sqrt_alphas_cumprod[t].view(b, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1, 1)
        return s1 * x0 + s2 * noise

    def p_mean_variance(self, model, x_t: torch.Tensor, t: torch.Tensor):
        """
        模型预测 eps, 由 eps 得到 x0，再算 posterior mean/var
        """
        eps_pred = model(x_t, t)
        b = x_t.shape[0]

        a_bar = self.alphas_cumprod[t].view(b, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
        x0_pred = x0_pred.clamp(-1, 1)

        coef1 = self.posterior_mean_coef1[t].view(b, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(b, 1, 1, 1)
        mean = coef1 * x0_pred + coef2 * x_t

        log_var = self.posterior_log_variance_clipped[t].view(b, 1, 1, 1)
        return mean, log_var

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t_int: int):
        b = x_t.shape[0]
        t = torch.full((b,), t_int, device=x_t.device, dtype=torch.long)
        mean, log_var = self.p_mean_variance(model, x_t, t)
        if t_int == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, model, n: int, image_size=32, channels=3):
        x = torch.randn((n, channels, image_size, image_size), device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x
        
    @torch.no_grad()
    def sample_ddim(self, model, n: int, steps: int = 50, eta: float = 0.0, image_size=32, channels=3):
        """
        DDIM sampling for speed. steps << T.
        eta=0 -> deterministic DDIM
        """
        x = torch.randn((n, channels, image_size, image_size), device=self.device)

        # 选取时间步序列（均匀抽样）
        step_indices = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
        step_indices = torch.unique(step_indices)  # 防止重复
        step_indices = step_indices.tolist()

        for i in range(len(step_indices)):
            t_int = step_indices[i]
            t_prev = step_indices[i + 1] if i + 1 < len(step_indices) else 0

            b = x.shape[0]
            t = torch.full((b,), t_int, device=self.device, dtype=torch.long)

            eps = model(x, t)

            a_t = self.alphas_cumprod[t_int]
            a_prev = self.alphas_cumprod[t_prev]

            a_t = a_t.clamp(1e-8, 1.0)
            a_prev = a_prev.clamp(1e-8, 1.0)

            # 预测 x0
            x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x0 = x0.clamp(-1, 1)

            # DDIM 参数
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0.0)
            sigma = sigma.view(1, 1, 1, 1)

            # direction term
            c = torch.sqrt((1 - a_prev) - sigma.pow(2)).clamp(min=0.0)
            c = c.view(1, 1, 1, 1)

            mean = torch.sqrt(a_prev).view(1, 1, 1, 1) * x0 + c * eps
            if t_int == 0:
                x = mean
            else:
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
                x = mean + sigma * noise

        return x

def ddpm_loss(ddpm: DDPM, model, x0: torch.Tensor):
    b = x0.shape[0]
    t = torch.randint(0, ddpm.T, (b,), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = ddpm.q_sample(x0, t, noise=noise)
    eps_pred = model(x_t, t)
    return F.mse_loss(eps_pred, noise)
