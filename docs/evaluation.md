# Evaluation methodology

Generative models have no single "accuracy". This project reports **five kinds of numbers**, and never mixes
them:

| Kind | Examples | Comparable across GAN and VAE? |
|---|---|---|
| **Model objective** | GAN D/G loss; VAE reconstruction, KL, −ELBO | **No.** They are different objectives on different scales |
| **Training diagnostics** | Mean D(x), D(G(z)) | No (GAN only) |
| **Proxy metrics** | Validity rate, class entropy, diversity ratios, FD-MNIST | Yes, with an identical procedure |
| **Distribution metric** | Standard FID | Yes, at equal sample counts |
| **Runtime** | Train and eval seconds, device | Context only |

A sixth category, **software tests** (pytest), checks code correctness. It is shown on a separate tab and
says nothing about model quality.

All values are computed from actual runs by `src/evaluation/evaluate.py` and stored in each experiment's
`metrics.json` / `metrics.csv`. Nothing is hand-entered.

## 1. GAN training diagnostics

* **D loss** = `BCE(D(x), 1−s) + BCE(D(G(z)), 0)` and **G loss** = `BCE(D(G(z)), 1)`, averaged per epoch
  (weighted by batch size).
* At the theoretical equilibrium D outputs 0.5 everywhere, so the D loss is near 2·ln 2 ≈ 1.386 (slightly
  different with label smoothing) and the G loss near ln 2 ≈ 0.693.
* **These are not quality metrics.** A falling G loss can mean better samples or a weaker discriminator.
  D(x) and D(G(z)) are mean sigmoid outputs and describe the discriminator's confidence. They are **not a
  classification accuracy**.

## 2. VAE reconstruction loss

Bernoulli negative log-likelihood: `BCE(logits, x)` **summed over the 784 pixels, then averaged over images**,
in nats per image. It is reported for training batches (an average over the epoch while the weights change)
and for the held-out validation split at the end of each epoch, with a fixed noise stream so epochs are
comparable. "BCE per pixel" is the same value ÷ 784, for readability.

## 3. KL divergence

`KL(q(z|x) ‖ N(0,I)) = −½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)`, summed over latent dimensions and averaged over
images (nats per image). The total objective is `recon + β·KL` (β = `kl_weight`, default 1).

## 4. Held-out digit classifier (the evaluator)

* Small CNN (2 conv blocks, 128-d penultimate layer, about 420k parameters).
* Trained only on the **training split**, for 3 epochs with Adam. The epoch with the best **validation**
  accuracy is kept.
* Its accuracy on the **official MNIST test set** (with per-class precision/recall/F1 and a confusion
  matrix from scikit-learn) is reported once. This number describes **the evaluator**, not the GAN or VAE.
* Cached at `outputs/evaluator/`. All experiments use the same evaluator.

## 5. Generated-sample validity

* `eval_num_samples` images (default 2,000) are sampled from the prior with seed `seed + 10000`.
* **Validity rate** = fraction of images whose highest softmax probability is **≥ `validity_threshold`**
  (default **0.9**).
* The same statistic is computed for an equal number of **real validation images**. That real-data value is
  the practical ceiling and shows how strict the threshold is.
* **Class entropy (normalised)** = entropy of the predicted-class histogram ÷ ln 10. Real MNIST scores about
  1.0. Low values indicate mode collapse. The full histogram is stored too.
* *What it measures:* "a classifier trained on real digits recognises this image confidently as *some*
  digit". It does **not** measure realism beyond what that classifier is sensitive to, and a generator can
  game it with prototypical digits that lack diversity, so read it together with the diversity metrics.

## 6. Diversity

* **Mean pairwise L2 distance** over all pairs of generated samples, computed exactly and deterministically
  given the sampling seed, in two spaces:
  * pixel space (`[0,1]`, 784-d);
  * classifier-feature space (128-d penultimate activations).
* Each is divided by the same statistic on an equally sized set of real validation images: the **diversity
  ratio**. A ratio near 1 means generated samples are as spread out as real data. Well below 1 suggests
  mode collapse or overly average samples. Above 1 can indicate noisy or broken samples. It is not
  automatically better.

## 7. FID (standard) and FD-MNIST (labelled alternative)

**FID.** The Fréchet distance between Gaussians fitted to Inception-v3 pool3 (2048-d) features of generated
and real images: `‖μ_g − μ_r‖² + Tr(Σ_g + Σ_r − 2(Σ_g Σ_r)^{1/2})`.

* Implementation: `pytorch-fid`'s `InceptionV3` with the **original TF FID weights** (downloaded once,
  ~92 MB). Images are replicated to 3 channels and bilinearly resized to 299×299. The matrix square root
  uses `scipy.linalg.sqrtm`, with the same numerical safeguards as the reference implementation.
* Reference set: the first `fid_num_samples` (default **1,000**) **validation** images. Generated set: the
  first `fid_num_samples` evaluation samples. Real-reference statistics are cached.
* **Limitations, which matter here:**
  * Inception was trained on ImageNet photos. MNIST digits are far out of distribution, so the features
    respond mainly to stroke texture and blur rather than to digit identity.
  * FID is **biased upward at small N**, and with N < 2048 the covariance matrices are rank-deficient (SciPy
    may warn that the matrix is singular). Values are valid for **comparisons at the same N within this
    project** and should not be compared with published MNIST FIDs, which use different N, reference sets
    and resizing.
  * On CPU, Inception is slow. Measured on the development laptop: about 4–7 images/s when idle and
    1.0–2.5 images/s under heavy background load. That cost is why N defaults to 1,000, the real-reference
    statistics are cached, and FID can be switched off.
  * If the weights cannot be downloaded, FID is recorded as `unavailable` with the reason, and nothing is
    substituted under the FID name.

**FD-MNIST (not standard FID).** The same Fréchet formula on the evaluator classifier's 128-d features,
using all `eval_num_samples` images on each side. It is domain-appropriate and cheap, but it depends on this
project's classifier, so it is **only comparable within this project**. It is always computed, and is the
labelled fallback when FID is unavailable.

## 8. Runtime

`train_seconds` (wall-clock time for all epochs, accumulated across resumes), `eval_seconds` (the whole
evaluation step, including FID), the device type and a device description (CPU model and thread count, or
GPU name) are stored in `metrics.json` and `metadata.json`. Runtime depends on hardware, background load,
epochs and data size. It is context, not a quality measure.

## 9. Experiment configuration and metadata

`metadata.json` stores: experiment ID, model name, the full config (seed, epochs, batch size, latent dim,
learning rate, betas, label smoothing / KL weight, evaluation settings), dataset sizes and split seed,
`test_set_used_for_training: false`, preprocessing, optimiser, Python/torch/torchvision/NumPy versions,
platform, CUDA availability and version, device and runtimes.

## Comparing GAN and VAE fairly

* Use the **same** epochs, `train_limit`, seed, `eval_num_samples`, `fid_num_samples` and validity
  threshold. The comparison page warns when these differ.
* Compare **only** validity, class entropy, diversity ratios, FD-MNIST, FID and runtime across models.
  Each has a stated direction: higher, lower, or closer to 1.
* Never compare a GAN loss with a VAE loss.
* A single run per model gives no estimate of seed variance. Treat small differences as inconclusive.
