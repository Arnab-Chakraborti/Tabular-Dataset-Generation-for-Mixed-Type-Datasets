"""
Loss functions for WGAN-GP with CTGAN-style conditional training.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Wasserstein Losses
# ==========================================================

def critic_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Critic loss.

    Minimize:
        E(fake) - E(real)
    """

    return fake_scores.mean() - real_scores.mean()


def generator_loss(
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Generator loss.

    Minimize:
        -E(fake)
    """

    return -fake_scores.mean()


# ==========================================================
# Gradient Penalty
# ==========================================================

def gradient_penalty(
    critic,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    cond: torch.Tensor | None = None,
):
    """
    Compute WGAN-GP gradient penalty.
    """

    device = real_samples.device

    batch_size = real_samples.size(0)

    alpha = torch.rand(
        batch_size,
        1,
        device=device,
    )

    alpha = alpha.expand_as(real_samples)

    interpolates = (
        alpha * real_samples
        + (1 - alpha) * fake_samples
    )

    interpolates.requires_grad_(True)

    critic_scores = critic(
        interpolates,
        cond,
    )

    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(
        batch_size,
        -1,
    )

    penalty = (
        gradients.norm(2, dim=1) - 1
    ) ** 2

    return penalty.mean()


# ==========================================================
# Conditional Loss
# ==========================================================

def conditional_loss(
    logits: List[torch.Tensor],
    cond: torch.Tensor,
    discrete_columns,
):
    """
    Cross-entropy loss for categorical columns.
    """

    if cond is None:
        return torch.tensor(
            0.0,
            device=logits[0].device,
        )

    losses = []

    for i, column in enumerate(discrete_columns):

        cond_slice = cond[
            :,
            column["cond_start"]:column["cond_end"],
        ]

        target = torch.argmax(
            cond_slice,
            dim=1,
        )

        loss = F.cross_entropy(
            logits[i],
            target,
        )

        losses.append(loss)

    return torch.stack(losses).mean()