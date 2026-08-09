---
layout: single
title: "Double-Clip MGDA: Clipping Away Double Sampling"
date: 2026-08-08 12:00:00 -0700
permalink: /blog/double-clip-mgda/
excerpt: "A direct walkthrough of Double-Clip MGDA, its primal-only form, and the surprising reason clipping can replace double sampling in stochastic multi-objective optimization."
categories:
  - Optimization
  - Machine Learning
tags:
  - multi-objective optimization
  - distributionally robust optimization
  - gradient clipping
author_profile: true
read_time: true
share: true
related: false
mathjax: true
featured_blog: true
header:
  teaser: /assets/images/double-clip-mgda-social.jpg
  og_image: /assets/images/double-clip-mgda-social.jpg
---

<figure class="dc-hero">
  <img src="{{ '/assets/images/double-clip-mgda-social.jpg' | relative_url }}" width="1200" height="630" decoding="async" alt="Double-Clip MGDA: balanced gradients pass through two clipping gates before branching to model and preference updates.">
</figure>

<p class="dc-lede">The main idea is unexpectedly simple: clip the <em>balanced</em> gradients, then reuse the same clipping factors in the model, dual, and preference updates. That coupling is what turns a difficult distributionally robust MOO procedure into a single-loop method—and what lets the analysis dispense with double sampling.</p>

<div class="dc-paper-links">
  <span>From our paper</span>
  <a href="https://arxiv.org/abs/2605.05660">arXiv abstract</a>
  <a href="https://arxiv.org/pdf/2605.05660">PDF</a>
</div>

## Algorithm 2, first

Consider $m$ robust objectives. The algorithm maintains three objects:

- $\theta_t$: the shared model parameters;
- $\eta_t$: one dual variable per objective, introduced by the distributionally robust reformulation;
- $w_t \in \mathcal W$: a preference vector on the probability simplex, so $w_t \ge 0$ and $\mathbf 1^\top w_t=1$.

Let $\widehat{\mathcal L}(\theta,\eta)$ denote the paper's rescaled vector of dual objectives. At iteration $t$, $Z_t$ collects its stochastic gradients with respect to the dual variables and $X_t$ collects its stochastic gradients with respect to the model. Algorithm 2 is:

<div class="paper-algorithm" role="group" aria-label="Algorithm 2: Double-Clip MGDA for DR-MOO">
  <div class="paper-algorithm__title"><strong>Algorithm 2:</strong> Double-Clip MGDA for DR-MOO</div>
  <div class="paper-algorithm__body">
    <div class="paper-algorithm__row"><span>1:</span><div>Initialize $\theta_0$, $\eta_0$, $w_0$, $\rho$, $\beta$, $\gamma$.</div></div>
    <div class="paper-algorithm__row"><span>2:</span><div><strong>Clipping rule:</strong> $\alpha_t=\min\!\left\{c_1,\frac{c_2}{\lVert X_tw_t\rVert}\right\}$, $\mu_t=\min\!\left\{f_1,\frac{f_2}{\lVert Z_tw_t\rVert}\right\}$.</div></div>
    <div class="paper-algorithm__row"><span>3:</span><div><strong>for</strong> $t=0,\ldots,T-1$ <strong>do</strong></div></div>
    <div class="paper-algorithm__row paper-algorithm__row--indent"><span>4:</span><div>Evaluate $Z_t=\nabla_\eta\widehat{\mathcal L}(\theta_t,\eta_t;\{\xi_t\}_B)$ with $B=N_2$.</div></div>
    <div class="paper-algorithm__row paper-algorithm__row--indent"><span>5:</span><div>$\eta_{t+1}=\eta_t-\gamma\mu_tZ_tw_t$.</div></div>
    <div class="paper-algorithm__row paper-algorithm__row--indent"><span>6:</span><div>Evaluate $X_t=\nabla_\theta\widehat{\mathcal L}(\theta_t,\eta_{t+1};\{\bar\xi_t\}_B)$ with $B=N_1$.</div></div>
    <div class="paper-algorithm__row paper-algorithm__row--indent"><span>7:</span><div>$\theta_{t+1}=\theta_t-\gamma\alpha_tX_tw_t$.</div></div>
    <div class="paper-algorithm__row paper-algorithm__row--indent"><span>8:</span><div>$w_{t+1}=\Pi_{\mathcal W}\!\left[w_t-\beta\!\left(\alpha_tX_t^\top X_tw_t+\mu_tZ_t^\top Z_tw_t+\rho w_t\right)\right]$.</div></div>
    <div class="paper-algorithm__row"><span>9:</span><div><strong>end for</strong></div></div>
  </div>
</div>

The order is deliberate. Here is what each block does.

### Lines 4–5: update the dual variable first

The dual variable $\eta_i$ is the finite-dimensional stand-in for the worst-case distribution of objective $i$. Line 4 estimates all dual gradients at the current pair $(\theta_t,\eta_t)$. Multiplying by $w_t$ forms the *balanced dual direction* $Z_tw_t$, and $\mu_t$ clips that direction before the step:

$$
\lVert \gamma \mu_t Z_tw_t\rVert \le \gamma f_2.
$$

This is not an inner optimization loop. It is one dual step per outer iteration. Objectives with larger current preference receive proportionally more influence through $w_t$, while clipping prevents an unstable dual direction from dominating the dynamics.

### Lines 6–7: use the new dual state to update the model

Only after computing $\eta_{t+1}$ do we evaluate $X_t$. In other words, the model gradient is measured at $(\theta_t,\eta_{t+1})$, not at the stale dual state. This sequential, Gauss-Seidel-like order is important: the model immediately responds to the latest approximation of each objective's adverse distribution.

The balanced model direction is $X_tw_t$. Its clipping factor $\alpha_t$ gives the analogous bound

$$
\lVert \gamma \alpha_t X_tw_t\rVert \le \gamma c_2.
$$

Thus both sides of the ill-conditioned geometry—the dual variables and the model parameters—move on controlled scales.

### Line 8: make the preference update obey the same geometry

Ordinary MGDA chooses $w$ to make the balanced gradient small. A projected gradient step for the model-gradient term naturally contains $X_t^\top X_tw_t$. In DR-MOO, stationarity also needs the dual component, which contributes $Z_t^\top Z_tw_t$.

The crucial detail is that line 8 does **not** use the raw Gram terms. It uses the same $\alpha_t$ and $\mu_t$ that controlled the parameter steps:

$$
\underbrace{\alpha_tX_t^\top X_tw_t}_{\text{model balance}}
\;+
\underbrace{\mu_tZ_t^\top Z_tw_t}_{\text{dual balance}}
\;+
\underbrace{\rho w_t}_{\text{regularization}}.
$$

Finally, $\Pi_{\mathcal W}$ projects the result back to the simplex, keeping the preferences nonnegative and summing to one. This shared scaling is the heart of Double-Clip MGDA: the preference vector is trained using the same bounded geometry that actually moves $\theta$ and $\eta$.

## The generic form: no dual variable required

The mechanism is not restricted to DRO. Suppose we have ordinary stochastic objectives collected in the vector $\mathbf F(\theta)$, and let

$$
X_t=[g_{1,t},\ldots,g_{m,t}] \in \mathbb R^{d\times m}
$$

be their stochastic gradient matrix. Starting from Algorithm 2, delete the dual variable $\eta_t$, the dual-gradient matrix $Z_t$, the clipping factor $\mu_t$, and every update involving them. Nothing else needs to change:

<div class="paper-algorithm" role="group" aria-label="Generic Double-Clip MGDA without dual variables">
  <div class="paper-algorithm__title"><strong>Generic framework:</strong> Double-Clip MGDA without dual variables</div>
  <div class="paper-algorithm__body">
    <div class="paper-algorithm__row"><span>1:</span><div>Initialize $\theta_0$, $w_0$, $\rho$, $\beta$, $\gamma$.</div></div>
    <div class="paper-algorithm__row"><span>2:</span><div><strong>Clipping rule:</strong> $\alpha_t=\min\!\left\{c_1,\frac{c_2}{\lVert X_tw_t\rVert}\right\}$.</div></div>
    <div class="paper-algorithm__row"><span>3:</span><div><strong>for</strong> $t=0,\ldots,T-1$ <strong>do</strong></div></div>
    <div class="paper-algorithm__row paper-algorithm__row--indent"><span>4:</span><div>Evaluate $X_t=\nabla_\theta\widehat{\mathbf F}(\theta_t;\{\xi_t\}_B)$ with $B=N_1$.</div></div>
    <div class="paper-algorithm__row paper-algorithm__row--indent"><span>5:</span><div>$\theta_{t+1}=\theta_t-\gamma\alpha_tX_tw_t$.</div></div>
    <div class="paper-algorithm__row paper-algorithm__row--indent"><span>6:</span><div>$w_{t+1}=\Pi_{\mathcal W}\!\left[w_t-\beta\!\left(\alpha_tX_t^\top X_tw_t+\rho w_t\right)\right]$.</div></div>
    <div class="paper-algorithm__row"><span>7:</span><div><strong>end for</strong></div></div>
  </div>
</div>

When $X_tw_t=0$, take $\alpha_t=c_1$. The same balanced-gradient clip is used twice: once in the model update and once in the preference-vector update. This primal-only framework is therefore Algorithm 2 with all dual-variable components removed.

## Proof intuition: why coupled clipping works

The usual stochastic MGDA preference step contains a one-batch Gram term such as $X_t^\top X_tw_t$. Because $\mathbb E[X_t^\top X_t]\ne(\mathbb E X_t)^\top(\mathbb E X_t)$, a standard bias-control device forms the product from two independent gradient batches. Double-Clip MGDA keeps one Gram estimate and controls its contribution instead. The proof is easiest to understand through two ideas.

### 1. Decouple problem constants from the clipping mechanism

The caps $c_1,f_1$ and radii $c_2,f_2$ are explicit algorithmic controls; they are not hidden inside smoothness, variance, or gradient-bound constants. Because

$$
\alpha_t\lVert X_tw_t\rVert\le c_2,
\qquad
\mu_t\lVert Z_tw_t\rVert\le f_2,
$$

the model and dual steps are bounded by $\gamma c_2$ and $\gamma f_2$. Reusing those same factors in the $w_t$ update also keeps the parameter dynamics and preference-vector dynamics on comparable scales. In Theorem 5.2, the choice $c_1=f_1=1/2$ and $c_2=f_2=\delta\epsilon$ makes the target accuracy enter through the clipping radii while allowing $\beta$ and $\gamma$ to remain order-one. This separation is what makes the magnitude controllable across $\theta_t$, $\eta_t$, and $w_t$.

### 2. Prove descent in the metric created by clipping

Expanding the projected preference update, as in Eq. (85) of Lemma L.2, produces the useful inner products

$$
-2\beta\alpha_t\langle w_t-w,X_t^\top X_tw_t\rangle
-2\beta\mu_t\langle w_t-w,Z_t^\top Z_tw_t\rangle,
$$

along with quadratic remainder terms such as

$$
\beta^2\alpha_t^2\lVert X_t^\top X_tw_t\rVert^2,
\qquad
\beta^2\mu_t^2\lVert Z_t^\top Z_tw_t\rVert^2.
$$

The normalization in $\alpha_t$ and $\mu_t$ is exactly what controls these remainders. Schematically,

$$
\alpha_t^2\lVert X_t^\top X_tw_t\rVert^2
\le c_2^2\lVert X_t\rVert_F^2,
\qquad
\mu_t^2\lVert Z_t^\top Z_tw_t\rVert^2
\le f_2^2\lVert Z_t\rVert_F^2.
$$

After scaling Eq. (85) and combining it with the sequential trajectory

$$
(\theta_t,\eta_t)\longrightarrow(\theta_t,\eta_{t+1})
\longrightarrow(\theta_{t+1},\eta_{t+1}),
$$

the analysis does not try to isolate a raw population quantity like $\lVert\nabla\widehat{\mathcal L}\,w_t\rVert^2$. Instead, it obtains descent directly in the clipped metric

$$
\gamma\alpha_t\lVert X_tw_t\rVert^2
+\gamma\mu_t\lVert Z_tw_t\rVert^2.
$$

This is the key match: the proof measures progress in the same geometry that the algorithm actually uses.

For the stochastic cross terms, write

$$
\widehat\Gamma_t=X_t-\nabla_\theta\widehat{\mathcal L}(\theta_t,\eta_{t+1}),
\qquad
\widehat\Upsilon_t=Z_t-\nabla_\eta\widehat{\mathcal L}(\theta_t,\eta_t).
$$

Adding and subtracting these population gradients decomposes the mixed terms in Eqs. (89)–(90). Cauchy–Schwarz then leaves expressions controlled directly by $\alpha_t\lVert X_tw_t\rVert$ or $\mu_t\lVert Z_tw_t\rVert$. Since $c_2=f_2=\delta\epsilon$, the preference-update remainders acquire an explicit $\delta^2\epsilon^2$ factor. They can be absorbed into the clipped descent bound without an independent Gram estimator and without forcing $\beta=O(\epsilon^2)$.

That is why this is a genuinely multi-objective argument. In a single-objective method there is no preference-vector trajectory, so neither the $w_t$ coupling terms nor this clipped preference descent exists. The result depends on clipping all three coupled dynamics—model, dual, and preference—not on importing a single-objective clipping proof.

<div class="dc-note">
  <strong>What “no double sampling” means here.</strong>
  Algorithm 2 still uses a dual-gradient batch and then a fresh model-gradient batch because the updates are sequential. What disappears is the need for two independent copies of the same gradient matrix solely to form an unbiased Gram product in the preference update.
</div>

Under the paper's assumptions, this produces a single-loop method with $O(\epsilon^{-4})$ sample complexity, compared with $O(\epsilon^{-12})$ for the double-loop DR-MOO baseline. The theory uses batches of order $O(\epsilon^{-2})$; in the reported ablation, a batch size of 256 was already sufficient for stable behavior.

## Experimental results

We used a pretrained ResNet-18 encoder with task-specific MLP heads, cross-entropy loss, and the dual of a $\chi^2$-divergence robust objective. Every baseline optimized the same dual DR-MOO formulation, with its hyperparameters tuned.

### Multi-MNIST: test accuracy under FGSM attack

The table below reports the accuracy values from Table 1 of the paper without additional aggregation or interpretation.

<div class="dc-table-wrap">
  <table class="dc-results-table dc-results-table--wide">
    <caption>Table 1: Test Accuracy under FGSM attack (%)</caption>
    <thead>
      <tr>
        <th rowspan="2">Method / Attack Level</th>
        <th colspan="5">Multi-MNIST 2-digits (70-epochs training)</th>
        <th colspan="5">Multi-MNIST 3-digits (100-epochs training)</th>
      </tr>
      <tr>
        <th>0.00</th><th>0.01</th><th>0.03</th><th>0.05</th><th>0.08</th>
        <th>0.00</th><th>0.01</th><th>0.03</th><th>0.05</th><th>0.08</th>
      </tr>
    </thead>
    <tbody>
      <tr><th><strong>Double-Clip MGDA</strong></th><td><strong>95.66%</strong></td><td><strong>83.48%</strong></td><td><strong>65.95%</strong></td><td><strong>60.40%</strong></td><td><strong>57.13%</strong></td><td><strong>98.76%</strong></td><td><strong>97.59%</strong></td><td><strong>94.40%</strong></td><td><strong>91.05%</strong></td><td><strong>86.65%</strong></td></tr>
      <tr><th>Double-loop MGDA</th><td>92.80%</td><td>72.81%</td><td>57.71%</td><td>54.49%</td><td>51.63%</td><td>97.49%</td><td>95.38%</td><td>89.99%</td><td>85.25%</td><td>79.88%</td></tr>
      <tr><th>MoCo</th><td>94.49%</td><td>77.69%</td><td>61.74%</td><td>58.63%</td><td>56.43%</td><td>98.27%</td><td>96.62%</td><td>92.75%</td><td>88.75%</td><td>83.43%</td></tr>
      <tr><th>NashMTL</th><td>91.21%</td><td>62.58%</td><td>51.67%</td><td>49.54%</td><td>47.09%</td><td>96.17%</td><td>92.31%</td><td>84.53%</td><td>78.91%</td><td>73.48%</td></tr>
      <tr><th>FAMO</th><td>89.05%</td><td>61.04%</td><td>50.82%</td><td>48.66%</td><td>46.48%</td><td>95.90%</td><td>91.86%</td><td>84.29%</td><td>78.94%</td><td>73.52%</td></tr>
      <tr><th>SDMGrad</th><td>89.59%</td><td>64.02%</td><td>52.00%</td><td>49.91%</td><td>47.31%</td><td>96.46%</td><td>92.89%</td><td>85.06%</td><td>79.37%</td><td>73.50%</td></tr>
      <tr><th>MoDo</th><td>91.10%</td><td>64.08%</td><td>51.95%</td><td>49.73%</td><td>47.49%</td><td>96.50%</td><td>93.15%</td><td>86.03%</td><td>80.56%</td><td>74.81%</td></tr>
      <tr><th>MGDA</th><td>89.44%</td><td>62.61%</td><td>52.19%</td><td>50.81%</td><td>48.30%</td><td>96.34%</td><td>92.85%</td><td>85.42%</td><td>80.26%</td><td>75.18%</td></tr>
    </tbody>
  </table>
</div>

### CelebA: robustness to task-wise label imbalance

Double-Clip MGDA also led every aggregate metric on CelebA. All entries are percentages averaged across tasks.

<div class="dc-table-wrap">
  <table class="dc-results-table">
    <caption>CelebA test performance</caption>
    <thead><tr><th>Method</th><th>Average accuracy</th><th>Balanced accuracy</th><th>AUC</th></tr></thead>
    <tbody>
      <tr class="dc-results-table__primary"><th>Double-Clip MGDA</th><td>88.41</td><td>89.55</td><td>94.63</td></tr>
      <tr><th>Double-Loop MGDA</th><td>86.61</td><td>87.97</td><td>93.28</td></tr>
      <tr><th>MoCo</th><td>87.27</td><td>88.50</td><td>93.31</td></tr>
      <tr><th>NashMTL</th><td>85.58</td><td>87.00</td><td>92.41</td></tr>
      <tr><th>FAMO</th><td>85.32</td><td>86.03</td><td>91.41</td></tr>
      <tr><th>SDMGrad</th><td>85.89</td><td>87.06</td><td>92.45</td></tr>
      <tr><th>MoDo</th><td>85.66</td><td>87.08</td><td>92.42</td></tr>
      <tr><th>MGDA</th><td>85.53</td><td>86.87</td><td>92.58</td></tr>
    </tbody>
  </table>
</div>

### Optimization and ablation evidence

The synthetic linear-regression and white-wine logistic-regression studies tell a complementary story: both proposed methods were competitive in balanced-gradient convergence, while Double-Clip avoided the expensive inner loop. The batch-size ablation also showed the practical trade-off predicted by the theory—small batches fluctuated, and the instability became negligible around batch size 256.

## More benchmarks are coming

These results are the first benchmark suite, not the last. We plan to add broader datasets, architectures, objectives, and stronger MOO baselines as we continue studying the algorithm. The especially interesting question is whether the primal-only form above can provide the same stability benefits outside distributionally robust objectives.

If you use Double-Clip MGDA—or build a new method from its balanced-gradient clipping idea—please cite our paper:

<div class="dc-citation" markdown="1">
**Yufeng Yang, Fangning Zhuo, Ziyi Chen, Heng Huang, and Yi Zhou.**<br>
“Distributionally Robust Multi-Objective Optimization.” arXiv:2605.05660, 2026.<br>
[Paper](https://arxiv.org/abs/2605.05660) · [DOI](https://doi.org/10.48550/arXiv.2605.05660)
</div>

```bibtex
@misc{yang2026distributionallyrobustmoo,
  title         = {Distributionally Robust Multi-Objective Optimization},
  author        = {Yang, Yufeng and Zhuo, Fangning and Chen, Ziyi and
                   Huang, Heng and Zhou, Yi},
  year          = {2026},
  eprint        = {2605.05660},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.05660}
}
```
