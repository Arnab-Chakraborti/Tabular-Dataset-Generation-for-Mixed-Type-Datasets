import torch


def run_latent_diagnostics(
    model,
    train_tensor,
    reconstruction_target=None,
    interpolation_steps=10,
    perturbation=0.10,
    verbose=True,
):
    """
    Complete latent-space and decoder diagnostics.

    Returns
    -------
    stats : dict
        Nested dictionary ready for MLFlow logging.
    """

    model.eval()

    if reconstruction_target is None:
        reconstruction_target = train_tensor

    stats = {}

    with torch.no_grad():

        ############################################################
        # ENCODER
        ############################################################

        mu, logvar = model.encoder(train_tensor)
        variance = torch.exp(logvar)

        latent = model.reparameterize(mu, logvar)

        prior = torch.randn_like(latent)

        ############################################################
        # BASIC
        ############################################################

        basic = {
            "mu_min": mu.min().item(),
            "mu_max": mu.max().item(),
            "mu_mean": mu.mean().item(),
            "var_min": variance.min().item(),
            "var_max": variance.max().item(),
            "var_mean": variance.mean().item(),
        }

        ############################################################
        # ACTIVE LATENT DIMS
        ############################################################

        latent_std = mu.std(dim=0)

        latent_stats = {
            "active_dimensions": int((latent_std > 0.10).sum()),
            "latent_dimensions": latent_std.numel(),
            "average_spread": latent_std.mean().item(),
            "posterior_mean": mu.mean().item(),
            "posterior_std": mu.std().item(),
            "prior_mean": prior.mean().item(),
            "prior_std": prior.std().item(),
            "posterior_variance_mean": variance.mean().item(),
        }

        ############################################################
        # KL
        ############################################################

        kl = -0.5 * (
            1
            + logvar
            - mu.pow(2)
            - variance
        )

        kl_dim = kl.mean(dim=0)

        kl_stats = {
            "average_kl": kl_dim.mean().item(),
            "dead_dimensions": int((kl_dim < 0.01).sum()),
        }

        ############################################################
        # CPU LINEAR ALGEBRA
        ############################################################

        mu_cpu = mu.detach().cpu()
        prior_cpu = prior.detach().cpu()

        cov = torch.cov(mu_cpu.T)

        eigvals = torch.linalg.eigvalsh(cov)

        effective_rank = int((eigvals > 1e-3).sum())

        condition_number = (
            eigvals.max() /
            (eigvals.min() + 1e-8)
        ).item()

        corr = torch.corrcoef(mu_cpu.T)

        off_diag = corr - torch.eye(corr.shape[0])

        covariance = {
            "effective_rank": effective_rank,
            "condition_number": condition_number,
            "mean_abs_corr": off_diag.abs().mean().item(),
            "max_abs_corr": off_diag.abs().max().item(),
            "smallest_eigenvalue": eigvals.min().item(),
            "largest_eigenvalue": eigvals.max().item(),
        }

        ############################################################
        # NORMS
        ############################################################

        posterior_norm = torch.norm(mu_cpu, dim=1)
        prior_norm = torch.norm(prior_cpu, dim=1)

        norm_stats = {
            "posterior_norm_mean": posterior_norm.mean().item(),
            "posterior_norm_std": posterior_norm.std().item(),
            "prior_norm_mean": prior_norm.mean().item(),
            "prior_norm_std": prior_norm.std().item(),
        }

        ############################################################
        # DISTANCES
        ############################################################

        sample = mu_cpu[:512]

        dist = torch.cdist(sample, sample)

        dist.fill_diagonal_(float("inf"))

        nearest = dist.min(dim=1).values

        distance_stats = {
            "nn_min": nearest.min().item(),
            "nn_mean": nearest.mean().item(),
            "nn_max": nearest.max().item(),
        }

        ############################################################
        # DECODER
        ############################################################

        recon = model.decoder(latent)

        prior_recon = model.decoder(prior)

        mse = torch.mean(
            (recon - reconstruction_target) ** 2
        ).item()
        target_variance = reconstruction_target.var().item()
        decoder_stats = {
            "posterior_recon_mse": mse,
            "target_variance": target_variance,
            "posterior_output_mean": recon.mean().item(),
            "posterior_output_std": recon.std().item(),
            "prior_output_mean": prior_recon.mean().item(),
            "prior_output_std": prior_recon.std().item(),
        }

        ############################################################
        # DECODER SENSITIVITY
        ############################################################

        eps = torch.randn_like(latent) * perturbation

        perturbed = model.decoder(latent + eps)

        sensitivity = (
            perturbed - recon
        ).abs().mean().item()

        decoder_stats["decoder_sensitivity"] = sensitivity

        ############################################################
        # INTERPOLATION
        ############################################################

        device = latent.device

        z_a = latent[0]
        z_b = latent[1]

        alpha = torch.linspace(
            0,
            1,
            interpolation_steps,
            device=device,
        )

        outputs = []

        for a in alpha:

            z = (1 - a) * z_a + a * z_b

            outputs.append(
                model.decoder(
                    z.unsqueeze(0)
                )
            )

        diffs = []

        for i in range(len(outputs) - 1):

            diffs.append(
                torch.mean(
                    torch.abs(
                        outputs[i + 1] - outputs[i]
                    )
                )
            )

        decoder_stats["interpolation_change"] = (
            torch.stack(diffs).mean().item()
        )

        ############################################################
        # JACOBIAN PROXY
        ############################################################

        delta = latent.clone()

        delta[:, 0] += perturbation

        changed = model.decoder(delta)

        jacobian_proxy = (
            changed - recon
        ).abs().mean().item()

        decoder_stats["jacobian_proxy"] = jacobian_proxy

    ############################################################
    # RETURN
    ############################################################

    stats["basic"] = basic
    stats["latent"] = latent_stats
    stats["kl"] = kl_stats
    stats["covariance"] = covariance
    stats["norm"] = norm_stats
    stats["distance"] = distance_stats
    stats["decoder"] = decoder_stats

    if verbose:

        print("\n================ LATENT DIAGNOSTICS ================\n")

        for section, values in stats.items():

            print(section.upper())

            for k, v in values.items():

                print(f"{k:30s}: {v}")

            print()

    return stats


def flatten_dict(d, parent_key=""):
    """
    MLFlow-friendly dictionary flattening.
    """

    items = {}

    for k, v in d.items():

        key = (
            f"{parent_key}.{k}"
            if parent_key
            else k
        )

        if isinstance(v, dict):

            items.update(flatten_dict(v, key))

        else:

            items[key] = float(v)

    return items
