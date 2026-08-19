"""
High-level health reports for latent space diagnostics.

This module intentionally contains NO PyTorch code.
It simply interprets statistics returned by diagnostics.latent.
"""

from typing import Dict, List


# ============================================================
# Utility
# ============================================================

def _status(ok: bool):
    return "🟢" if ok else "🟡"


# ============================================================
# Main Report Generator
# ============================================================

def generate_latent_report(stats: Dict) -> str:
    """
    Generates a concise human-readable report.

    Parameters
    ----------
    stats : dict
        Dictionary returned by diagnostics.latent.run()

    Returns
    -------
    str
        Multi-line report suitable for terminal and MLFlow.
    """

    report: List[str] = []

    report.append("")
    report.append("=" * 60)
    report.append("LATENT SPACE HEALTH REPORT")
    report.append("=" * 60)
    report.append("")

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    posterior_std = stats["latent"]["posterior_std"]

    encoder_ok = 0.75 <= posterior_std <= 1.25

    report.append(
        f"{_status(encoder_ok)} Encoder Distribution : "
        f"Posterior std = {posterior_std:.3f}"
    )

    if encoder_ok:
        report.append("    Posterior distribution is close to the unit Gaussian.")
    else:
        report.append("    Posterior is drifting away from the prior.")

    report.append("")

    # --------------------------------------------------------
    # Active Dimensions
    # --------------------------------------------------------

    active_dims = stats["latent"]["active_dimensions"]
    latent_dim = stats["latent"]["latent_dimensions"]

    active_ratio = active_dims / latent_dim

    active_ok = active_ratio > 0.80

    report.append(
        f"{_status(active_ok)} Latent Utilisation : "
        f"{active_dims}/{latent_dim} dimensions active"
    )

    if active_ok:
        report.append("    Most latent dimensions are contributing.")
    else:
        report.append("    Many latent dimensions are inactive.")

    report.append("")

    # --------------------------------------------------------
    # KL
    # --------------------------------------------------------

    dead_dims = stats["kl"]["dead_dimensions"]

    dead_ratio = dead_dims / latent_dim

    kl_ok = dead_ratio < 0.20

    report.append(
        f"{_status(kl_ok)} KL Health : "
        f"{dead_dims} dead dimensions"
    )

    if kl_ok:
        report.append("    KL regularisation appears balanced.")
    else:
        report.append("    Significant posterior collapse detected.")

    report.append("")

    # --------------------------------------------------------
    # Covariance
    # --------------------------------------------------------

    effective_rank = stats["covariance"]["effective_rank"]

    rank_ratio = effective_rank / latent_dim

    rank_ok = rank_ratio > 0.80

    report.append(
        f"{_status(rank_ok)} Latent Diversity : "
        f"Effective Rank = {effective_rank}"
    )

    if rank_ok:
        report.append("    Latent space spans most available dimensions.")
    else:
        report.append("    Latent space occupies a relatively small subspace.")

    report.append("")

    # --------------------------------------------------------
    # Condition Number
    # --------------------------------------------------------

    cond = stats["covariance"]["condition_number"]

    cond_ok = cond < 1e4

    report.append(
        f"{_status(cond_ok)} Numerical Conditioning : "
        f"Condition Number = {cond:.2f}"
    )

    if cond_ok:
        report.append("    Covariance matrix is well conditioned.")
    else:
        report.append("    Strong anisotropy detected in latent space.")

    report.append("")

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    corr = stats["covariance"]["max_abs_corr"]

    corr_ok = corr < 0.60

    report.append(
        f"{_status(corr_ok)} Latent Independence : "
        f"Max correlation = {corr:.3f}"
    )

    if corr_ok:
        report.append("    Latent dimensions remain reasonably independent.")
    else:
        report.append("    Several latent dimensions are strongly correlated.")

    report.append("")

    # --------------------------------------------------------
    # Decoder Reconstruction
    # --------------------------------------------------------

    recon = stats["decoder"]["posterior_recon_mse"]
    target_var = stats["decoder"]["target_variance"]

    # Normalized reconstruction error
    normalized_recon = recon / max(target_var, 1e-8)

    recon_ok = normalized_recon < 1.0

    report.append(
        f"{_status(recon_ok)} Decoder Reconstruction : "
        f"Normalized MSE = {normalized_recon:.3f}"
    )

    if normalized_recon < 0.25:
        report.append(
            "    Reconstruction error is much smaller than the natural variation "
            "present in the latent distribution."
        )
    elif normalized_recon < 1.0:
        report.append(
            "    Reconstruction error is smaller than the intrinsic data variance. "
            "The decoder is preserving most latent information."
        )
    elif normalized_recon < 2.0:
        report.append(
            "    Reconstruction error is comparable to the latent variance. "
            "Some information is being lost during decoding."
        )
    else:
        report.append(
            "    Reconstruction error exceeds the natural variation of the latent "
            "distribution. The decoder may be underfitting or unstable."
        )

    report.append("")

    # --------------------------------------------------------
    # Decoder Sensitivity
    # --------------------------------------------------------

    sensitivity = stats["decoder"]["decoder_sensitivity"]

    sens_ok = 0.01 < sensitivity < 0.20

    report.append(
        f"{_status(sens_ok)} Decoder Responsiveness : "
        f"{sensitivity:.5f}"
    )

    if sens_ok:
        report.append("    Decoder responds smoothly to latent perturbations.")
    else:
        report.append("    Decoder may be too insensitive or unstable.")

    report.append("")

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    score = sum([
        encoder_ok,
        active_ok,
        kl_ok,
        rank_ok,
        cond_ok,
        corr_ok,
        recon_ok,
        sens_ok,
    ])

    report.append("=" * 60)

    if score >= 7:
        verdict = "🟢 Overall Status : HEALTHY"
    elif score >= 5:
        verdict = "🟡 Overall Status : ACCEPTABLE"
    else:
        verdict = "🔴 Overall Status : NEEDS ATTENTION"

    report.append(verdict)
    report.append(f"Health Score : {score}/8")

    report.append("=" * 60)

    return "\n".join(report)
