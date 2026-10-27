# Methodology

## Why compare a GAN and a VAE?

GANs and VAEs are the two classic families of deep generative models. They learn the same thing, a
distribution over images that can be sampled, through opposite strategies:

| | GAN | VAE |
|---|---|---|
| Training signal | Adversarial: a discriminator learns to tell real from fake, and the generator learns to fool it | Likelihood-based: maximise a lower bound (ELBO) on log p(x) |
| Encoder | No | Yes (amortised inference q(z\|x)) |
| Typical samples | Sharp, but may cover only some modes | Smooth or blurry, but cover the data well |
| Training stability | Fragile (balance of two players, mode collapse) | Stable, monotone loss |
| Quality signal during training | None. Losses describe the game, not quality | Held-out ELBO is a meaningful, comparable number |

Comparing them on one dataset, with one evaluation protocol, shows the trade-offs a solution architect has
to explain when choosing a generative approach: fidelity versus coverage, stability, controllability, and
how hard each is to evaluate.

## Dataset

* **MNIST** (LeCun et al.): 60,000 training and 10,000 test grayscale 28×28 handwritten digits,
  downloaded automatically by `torchvision`.
* **Split policy**: the 60k official training images are permuted with a fixed seed (2024) into
  **55,000 training** and **5,000 validation** images. The official **test** set is kept separate.
  * GAN, VAE and the evaluator classifier update weights **only** on the training split.
  * The validation split is used for VAE held-out loss, for the classifier's model selection
    (best epoch), and as the real-data reference for all generative metrics.
  * The test set is read once, to report the evaluator classifier's accuracy. It plays no role in
    training the generative models, in choosing their hyperparameters, or in their metrics.
  * `train_limit` takes the first *N* indices of the shuffled training split, so quick runs use a
    subset of the full split and never touch validation images. `tests/test_data.py` checks this.
* **Preprocessing**: `x/255 → [0,1]`. The GAN uses `2x − 1 → [−1,1]` to match its Tanh output. The VAE
  keeps `[0,1]` because pixels are modelled as Bernoulli probabilities. No augmentation. All evaluation
  happens on `[0,1]` images.
* **Limitations**: MNIST is small, centred, clean and single-channel. A model that works here is not
  evidence that it works on natural images or production data. MNIST quality metrics saturate quickly.

## GAN

**Architecture (DCGAN-style).** Generator: `z ∈ ℝ^d → Linear → BN → ReLU → 128×7×7 → ConvT(4,2,1) →
BN → ReLU → 64×14×14 → ConvT(4,2,1) → Tanh → 1×28×28`. Discriminator: `Conv(4,2,1) → LeakyReLU(0.2) →
Conv(4,2,1) → BN → LeakyReLU → Linear → logit`. Weights are initialised from N(0, 0.02) (Radford et al., 2016).

**Objective.** With D outputting logits and σ the sigmoid:

* Discriminator: minimise `BCE(D(x), 1 − s) + BCE(D(G(z)), 0)`, where `s` is the one-sided label smoothing
  (default 0.1). This keeps D from becoming over-confident on real data.
* Generator: minimise the *non-saturating* loss `BCE(D(G(z)), 1) = −log σ(D(G(z)))`, which gives stronger
  gradients early in training than minimising `log(1 − D(G(z)))`.
* Optimiser: Adam, lr 2e-4, β = (0.5, 0.999). One D step per G step.

**Instability handling.** Label smoothing, BatchNorm and the DCGAN learning rate are standard stabilisers.
Every batch is checked for NaN/Inf. Divergence stops the run with a concrete suggestion (lower lr or more
smoothing). D(x) and D(G(z)) are logged every epoch. If D(x) → 1 and D(G(z)) → 0 for many epochs, the
discriminator is winning. A collapsing class histogram in evaluation (low normalised class entropy)
indicates mode collapse.

## VAE

**Architecture.** Encoder: `Conv(1→32,4,2,1) → ReLU → Conv(32→64,4,2,1) → ReLU → Linear(3136→256) → ReLU`,
followed by two heads for μ and log σ² (latent dim 16 by default). The decoder mirrors the encoder and
outputs logits.

**Objective.** The VAE minimises the negative evidence lower bound:

```
L(x) = E_{q(z|x)} [ −log p(x|z) ]  +  β · KL( q(z|x) ‖ N(0, I) )
```

* **Reparameterisation:** `z = μ + σ ⊙ ε, ε ~ N(0, I)`. Sampling becomes a deterministic function of
  (μ, σ) plus external noise, so gradients flow into the encoder.
* **Reconstruction:** p(x|z) is a product of 784 Bernoullis, so the term is binary cross-entropy (computed
  from logits for numerical stability), **summed over pixels and averaged over the batch** (nats per image).
* **KL (closed form):** `−½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)`, **summed over latent dims and averaged over
  the batch**. log σ² is clamped to [−10, 10] for numerical safety.
* β = 1 is the standard ELBO. β is configurable (β-VAE).
* Optimiser: Adam, lr 1e-3.

**Reconstruction-versus-regularisation trade-off.** The reconstruction term rewards encoding as much
information about x as possible. The KL term pulls every posterior toward the same prior, which limits how
much information z can carry and keeps the latent space smooth and sample-able.

* Small β, or a large latent dim: sharper reconstructions, but "holes" in latent space, so prior samples
  can look worse.
* Large β: smoother latent space and better-behaved prior samples, but blurrier reconstructions and
  possible **posterior collapse** (KL → 0, decoder ignores z).
* With a Bernoulli decoder, the optimum averages over plausible images. That is the main reason VAE
  samples look blurry compared with GAN samples.

## Reproducibility

* Every experiment stores its full configuration, including seed, split seed, and dataset size.
* `set_seed` seeds Python, NumPy and PyTorch, disables cuDNN autotuning and enables deterministic algorithms
  (`warn_only`, so an op without a deterministic kernel warns rather than crashing a long run).
* Separate RNG streams are used for: batch order (`seed + epoch`), the fixed noise for per-epoch grids
  (`seed + 1`), VAE validation noise (`seed + 2`) and evaluation sampling (`seed + 10000`). Evaluation
  therefore never perturbs training randomness.
* On CPU, identical seeds give identical losses, and a resumed run matches an uninterrupted one. Both are
  tested. On GPU, determinism is best-effort.

## Experiment protocol used for the reported results

1. Train the evaluator classifier once (`python cli.py train-classifier`).
2. Train each model with its default hyperparameters, the same seed, the same training data size and
   the same number of epochs.
3. Evaluate both with the same `eval_num_samples`, `fid_num_samples`, validity threshold and validation
   reference.
4. Compare only the metrics defined identically for both models (see [evaluation.md](evaluation.md)).

Hyperparameters were not tuned on any metric. The defaults come from the literature (DCGAN) and from
common practice (VAE).
