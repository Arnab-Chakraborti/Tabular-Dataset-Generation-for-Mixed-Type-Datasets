from dataclasses import dataclass


@dataclass
class GANConfig:

    # ------------------------
    # Reproducibility
    # ------------------------
    random_state: int = 42

    # ------------------------
    # Dataset
    # ------------------------
    batch_size: int = 512

    # ------------------------
    # BGMM
    # ------------------------
    max_components: int = 10
    weight_threshold: float = 1e-3
    weight_concentration_prior: float = 0.001
    bgmm_max_iter: int = 500

    # ------------------------
    # Generator
    # ------------------------
    latent_dim: int = 128
    generator_hidden_dims: tuple[int, ...] = (256, 256, 512)

    # ------------------------
    # Critic
    # ------------------------
    critic_hidden_dims: tuple[int, ...] = (512, 256, 256)
    critic_negative_slope: float = 0.2

    # ------------------------
    # Optimizer
    # ------------------------
    learning_rate: float = 1e-4
    beta1: float = 0.0
    beta2: float = 0.9

    # ------------------------
    # WGAN-GP
    # ------------------------
    lambda_gp: float = 10.0
    n_critic: int = 5

    # ------------------------
    # Training
    # ------------------------
    epochs: int = 300

    # ------------------------
    # Gumbel Softmax
    # ------------------------
    temperature: float = 0.2

 # ------------------------
    # Conditional Loss
    # ------------------------
    conditional_loss_weight: float = 1.0