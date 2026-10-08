# GNN
Non-linear Dynamics on Graphs
# A Three-Layer Learning Framework for Compositional Prediction of Dynamics in Large Gene Regulatory Networks

**Full Specification: Primary Architecture (Version 1) and Oscillatory Extension (Version 2)**

## Abstract

Large gene regulatory networks (GRNs) exhibit a variety of nonlinear dynamical behaviors. Empirical and computational studies of random GRN ensembles consistently show that the vast majority of biological GRNs under realistic transcription rules are dominated by fixed point attractors — monostability and multistability — while sustained oscillations (limit cycles) are concentrated in specific functional network types such as circadian clocks and cell cycle engines. Chaos has never been reported in real biological GRNs.

This document develops a machine learning architecture that predicts GRN dynamics compositionally by exploiting the SCC–DAG decomposition: every GRN decomposes into strongly connected components (SCCs) connected by a directed acyclic graph (DAG). All feedback is confined within SCCs; all inter-SCC connections are strictly feed-forward. The architecture predicts the dynamical repertoire of each SCC in isolation (Layer I), computes the inter-SCC regulatory signal analytically via the known Hill function (Layer II), and predicts the conditioned response of the receiving SCC (Layer III).

The document is organized in two parts reflecting the empirical frequency of dynamical behaviors in real GRNs.

**Part I: Primary Architecture (Version 1)** handles the dominant case: fixed point attractors only. Every SCC is characterized by its fixed points, their eigenspectra, and their stability. Every inter-SCC signal is a constant scalar derived from a stable fixed point value passed through the Hill function. This covers monostability, multistability, saddle points, and bifurcations between them. It is the architecture that should be built, validated, and published first.

**Part II: Oscillatory Extension (Version 2)** extends the primary architecture to handle limit cycles and periodic inter-SCC signals. This extension is mathematically valid but adds substantial complexity — forced nonautonomous dynamics, augmented state spaces, phase conventions, and regime-split physics losses — that is only warranted after Version 1 has been empirically validated.

---

## Part I — Primary Architecture: Fixed Point Dynamics (Version 1)

### Why Fixed Points Dominate Biological GRNs

Before developing the architecture it is worth understanding why fixed point dynamics is the right primary focus.

Li et al. (2013) studied the generic dynamical properties of random GRNs with continuous variables across a range of transcription regulatory rules. Their key findings are:

- Under biologically realistic transcription rules (AND rule, additive rule), the vast majority of GRN trajectories converge to fixed point attractors. The probability of limit cycle behavior drops quickly with increasing network connectivity, resulting in networks dominated by steady state behavior at high connectivity.

- Limit cycles, when they occur, are generated primarily by short negative feedback loops of length three to five. Shorter motifs with faster frequencies tend to override longer ones. The limit cycle is driven by a specific core motif, not the full network topology.

- Chaotic behavior has never been reported in real biological GRNs and appears only under biologically unrealistic strong inhibition rules at very high connectivity.

- Multistability — multiple coexisting fixed point attractors — is the dominant form of complex behavior and is central to cell fate decisions, differentiation, and development.

The biological roles of these behaviors are correspondingly tiered:


| **Behavior**   | **Biological role**                                | **Frequency**               |
|:---------------|:---------------------------------------------------|:----------------------------|
| Monostability  | Homeostasis, constitutive expression               | Dominant                    |
| Multistability | Cell fate decisions, differentiation               | Common                      |
| Saddle points  | Basin boundaries between cell fates                | Accompanies multistability  |
| Limit cycles   | Circadian clock, cell cycle, metabolic oscillators | Rare, functionally specific |
| Chaos          | Unknown in biology                                 | Never reported              |


###### Consequence for architecture design.

The primary architecture should handle fixed points — monostability, multistability, saddle points — with full rigor. Limit cycle support is a meaningful extension but should not be the primary design target. The oscillatory extension (Part II) is important for completeness but its empirical validation should follow, not precede, validation of the primary architecture.

### Dynamical Model

#### The GRN ODE

A gene regulatory network is a directed signed graph $G = (V, E)$ with $|V| = N$ genes. The expression level $x_i \geq 0$ of gene $i$ evolves as: $$\dot{x}_i = P_i \prod_{j \rightarrow i} H_{ji}(x_j) - K_i^{\deg} x_i,
\qquad i \in V.
$$ The interpretation: each gene has a basal production rate $P_i > 0$, modified by the product of Hill regulatory factors from all its upstream regulators, and degraded at rate $K_i^{\deg} > 0$ proportional to its current expression level.

#### Hill Regulation Functions

Each regulatory edge $j \to i$ contributes one Hill factor: $$H_{ji}(x_j;\, T_{ji}, n_{ji}, F_{ji})
= \frac{F_{ji}(x_j / T_{ji})^{n_{ji}} + 1}{(x_j / T_{ji})^{n_{ji}} + 1},
$$ where $T_{ji} > 0$ is the threshold concentration at which regulation is half-maximal, $n_{ji} \geq 1$ is the Hill coefficient controlling the steepness of the response (larger $n_{ji}$ gives sharper switch-like behavior), and $F_{ji} > 0$ is the fold-change: $F_{ji} > 1$ means activation (the factor increases production), $F_{ji} < 1$ means repression.

A key property is that the Hill function is bounded: $H_{ji} \in (\min(1, F_{ji}), \max(1, F_{ji}))$ for all finite $x_j > 0$. This boundedness is what makes the ODE biologically reasonable and mathematically tractable.

#### The Positively Invariant Box

Because each Hill factor is bounded above by $\max(1, F_{ji})$ and the production term is a product of Hill factors, the maximum possible production rate for gene $i$ is $P_i \prod_{j \to i} \max(1, F_{ji})$. We use a slightly enlarged version to ensure strict inwardness on all faces:


**Definition 1** (Positively invariant box). *Fix $\varepsilon > 0$ and define: $$\tilde{x}_i^{\max}
= (1 + \varepsilon) \frac{P_i}{K_i^{\deg}} \prod_{j \rightarrow i} \max(1, F_{ji}),
$$ with the empty product equal to $1$ for genes with no regulators. Define the box $\Omega = \{x \in \mathbb{R}^N : 0 \leq x_i \leq \tilde{x}_i^{\max},\; \forall i\}$.*



**Remark 1** (Why the product, not the max). *An earlier incorrect formula used $\max_j F_{ji}$ in place of $\prod_j \max(1, F_{ji})$. This is wrong: two activating regulators with fold-changes $F_1 = F_2 = 5$ can jointly produce a maximal Hill product approaching $25$, not $5$. The product bound is essential when multiple activating regulators combine.*



**Remark 2** (Positive invariance with strict inwardness). *On the lower face $x_i = 0$: $\dot{x}_i = P_i \prod_j H_{ji}(x_j) > 0$ since all Hill factors exceed $\min(1, F_{ji}) > 0$. Strictly inward.*

*On the upper face $x_i = \tilde{x}_i^{\max}$: $$\dot{x}_i
\leq P_i \prod_j \max(1, F_{ji}) - K_i^{\deg} \tilde{x}_i^{\max}
= -\varepsilon K_i^{\deg} \frac{P_i}{K_i^{\deg}} \prod_j \max(1, F_{ji}) < 0.$$ Strictly negative by the $(1+\varepsilon)$ factor. Therefore $\Omega$ is positively invariant with strict inwardness on all faces, meaning $f(x) \neq 0$ on $\partial\Omega$.*


#### Fixed Points and Their Local Characterization


**Definition 2** (Fixed point). *A fixed point is any $x^* \in \Omega$ satisfying $f(x^*) = 0$: $$P_i \prod_{j \rightarrow i} H_{ji}(x_j^*) = K_i^{\deg} x_i^*, \qquad \forall\, i \in V.
$$ Fixed points are not assumed stable. They include stable attractors, unstable repellers, and saddle points.*



**Definition 3** (Jacobian and stability classification). *At a fixed point $x^*$, the Jacobian matrix is: $$J_{ij}(x^*) = \frac{\partial \dot{x}_i}{\partial x_j}\bigg|_{x = x^*}.$$ The eigenvalues of $J(x^*)$ take the form $\lambda_k = \alpha_k \pm i\beta_k$. The stability class is determined by the signs of the real parts and the presence of nonzero imaginary parts:*


| *Real parts*         | *Imaginary parts*       | *Stability class*                                                                                                       |
|:---------------------|:------------------------|:------------------------------------------------------------------------------------------------------------------------|
| *All $\alpha_k < 0$* | *All $\beta_k = 0$*     | *Stable node: locally asymptotically stable, real spectrum. Individual components may overshoot in $n > 1$ dimensions.* |
| *All $\alpha_k < 0$* | *Some $\beta_k \neq 0$* | *Stable spiral: locally asymptotically stable, oscillatory approach*                                                    |
| *All $\alpha_k > 0$* | *All $\beta_k = 0$*     | *Unstable node: locally unstable, real spectrum*                                                                        |
| *All $\alpha_k > 0$* | *Some $\beta_k \neq 0$* | *Unstable spiral: locally unstable, oscillatory departure*                                                              |
| *Mixed signs*        | *Any*                   | *Saddle: stable manifold attracts in some directions, unstable manifold repels in others*                               |




**Remark 3** (Why unstable spirals are heuristic triggers, not limit cycle guarantees). *In two-dimensional systems, the Poincaré–Bendixson theorem guarantees that a bounded trajectory that cannot converge to a fixed point must approach a limit cycle. In $n \geq 3$ dimensions this theorem does not hold: a trajectory repelled from an unstable spiral could wind on a torus or exhibit other complex behavior. In the Version 1 architecture we treat unstable spirals as part of the fixed point catalog (they are fixed points with specific eigenspectrum signature) and do not attempt to predict limit cycles. The presence of an unstable spiral is noted in the phase portrait but does not trigger limit cycle search in Version 1.*


#### The Version 1 Phase Portrait

In Version 1, the phase portrait of an SCC contains only fixed point objects:


**Definition 4** (Version 1 phase portrait). *For a parameterized SCC realization $\mathcal{G}_i = (G_i, \vartheta_i)$, the Version 1 phase portrait is: $$\mathcal{P}_i
= \left\{
\left(x^{*(r)},\, \{\hat{c}_m^{(r)}\}_{m=1}^{n_i},\, \sigma^{(r)}
\right)
\right\}_{r \in \mathcal{I}_{\mathrm{FP}}},
$$ where $x^{*(r)} \in \mathbb{R}^{n_i}$ is the location of fixed point $r$, $\{\hat{c}_m^{(r)}\}_{m=1}^{n_i}$ are the coefficients of the characteristic polynomial of $J(x^{*(r)})$ (encoding the eigenspectrum in a permutation-invariant form), and $\sigma^{(r)}$ is the stability class derived from the characteristic polynomial.*


The characteristic polynomial coefficients $\{c_m\}$ are used in place of direct eigenvalue predictions because eigenvalues are unordered and form complex-conjugate pairs, making direct supervised matching ambiguous. The characteristic polynomial is permutation-invariant: its coefficients are elementary symmetric polynomials of the eigenvalues and are uniquely determined regardless of eigenvalue ordering.

### SCC Decomposition and Condensation DAG

<div class="proposition">

**Proposition 1** (Condensation DAG). *Collapsing each SCC of the GRN to a single meta-node and retaining the inter-SCC edges gives a directed acyclic graph (DAG). Every feedback loop of the GRN is confined within a single SCC. All inter-SCC edges are strictly feed-forward.*


This is an exact graph-theoretic fact, not an approximation. It is the structural foundation of the entire architecture: because feedback is local (inside SCCs) and inter-SCC flow is global (along the DAG), the dynamics of the whole network can be assembled compositionally from the dynamics of the individual SCCs.

<div id="ass:bounded" class="assumption">

**Assumption 1** (Bounded SCC size). *There exists $N_{\max}$ such that $|G_i| \leq N_{\max}$ for every SCC, with a practical working value of $N_{\max} \approx 8$–$10$ genes.*


This assumption is motivated by the biology: the functional modules of real GRNs (toggle switches, feed-forward loops, oscillator cores) are small. Under this assumption, every call to the encoder processes a graph of at most $N_{\max}$ nodes, regardless of the size of the full GRN.

### Tier 1 Algebraic Conditions

A central design principle is the two-tier pipeline. Tier 1 consists of cheap algebraic and topological conditions that can be evaluated before any numerical computation. They serve two roles: (1) as corpus generation gates that determine which numerical computations are needed, and (2) as physics-informed loss terms or evaluation metrics that enforce mathematical self-consistency on the model’s predictions.

#### Thomas Necessary Conditions

<div id="thm:thomas" class="theorem">

**Theorem 1** (Thomas necessary conditions, Version 1 relevant form). *For the GRN dynamics <a href="#eq:grn" data-reference-type="eqref" data-reference="eq:grn">[eq:grn]</a>:*

- ***Multistability requires positive feedback:** Multiple stable fixed points can exist only if the signed interaction graph contains at least one positive feedback loop (a directed cycle with an even number of repressive edges).*

*Contrapositive (the operationally useful form): $$\text{No positive feedback loop in } G
\;\Rightarrow\; \text{at most one stable fixed point.}
$$ This is a *necessary condition* only. Presence of positive feedback does not guarantee multistability; it only makes it possible.*


###### How this is used in Version 1.

Before any numerical computation, inspect the signed graph. If no positive feedback loop exists, the decoder is constrained to predict at most one stable FP record. Multistability search (multiple Newton–Raphson seeds) is still run but the output is constrained. This is an exclusion filter, not a classifier.


**Remark 4** (Why this is one-directional). *It would be tempting to write: “positive feedback present $\Rightarrow$ multistability.” This is false. A positive feedback loop is necessary but not sufficient for multistability; the kinetic parameters also must cooperate. The Thomas condition only tells us what is impossible (monostability when positive feedback is absent), never what is guaranteed.*


#### Brouwer Degree Constraint

<div id="thm:brouwer" class="theorem">

**Theorem 2** (Brouwer degree on $\Omega$). *Let $g(x) = -f(x)$ be the negated GRN vector field. Since $f$ points strictly inward on all faces of $\partial\Omega$ (established in Definition <a href="#def:box" data-reference-type="ref" data-reference="def:box">1</a> via the $(1+\varepsilon)$ factor), $g$ points strictly outward on $\partial\Omega$. Therefore the Brouwer degree of $g$ with respect to the origin on $\Omega$ equals $1$. For isolated hyperbolic fixed points $\{x^{*(r)}\} \subset \Omega$: $$\boxed{
\sum_r \mathrm{ind}_g(x^{*(r)}) = 1,
\qquad
\mathrm{ind}_g(x^*) = \operatorname{sgn}(\det(-J(x^*))).
}
$$*



**Remark 5** (Index formula and complex eigenvalue pairs). *The determinant $\det(J) = \prod_k \lambda_k$. A complex conjugate pair $\alpha \pm i\beta$ contributes $(\alpha + i\beta)(\alpha - i\beta) =
\alpha^2 + \beta^2 > 0$ to the product, always positive regardless of the sign of $\alpha$. Therefore complex conjugate pairs (stable or unstable spirals) never change the sign of $\det(J)$ and never affect the index. Only real eigenvalues affect $\operatorname{sgn}(\det(J))$. The index formula $\operatorname{sgn}(\det(-J))$ is unambiguous and handles all cases.*


###### Practical consequence: bistability requires an odd-index saddle.

A system with two stable fixed points (each with index $(-1)^n \cdot (+1)$ since all eigenvalues have negative real parts making $\det(J) > 0$) must have remaining fixed points whose indices sum to compensate. This means at least one saddle (mixed eigenvalue signs) must exist between the two stable attractors. The Brouwer constraint is a free consistency check that tells us when Newton–Raphson has found all fixed points.

###### Use as corpus completeness check and evaluation metric only.

$\operatorname{sgn}(\det(J))$ is piecewise constant and has zero gradient almost everywhere. It cannot be used as a differentiable training loss. We use it only as (1) a corpus generation completeness check: if the found fixed points do not satisfy $\sum_r \mathrm{ind}(x^{*(r)}) = 1$, more seeds are needed; and (2) a test-time evaluation metric: the fraction of predicted fixed point sets satisfying the constraint.

#### Hopf Proximity Indicator (Version 1: No BVP Triggered)

A nondegenerate Hopf bifurcation requires: a simple complex conjugate eigenvalue pair $\pm i\omega_0$ with no other imaginary eigenvalues, a transversality condition, and a nonzero first Lyapunov coefficient.

In Version 1, we do not trigger BVP continuation for limit cycles. However, for every fixed point that has at least one complex conjugate eigenvalue pair $(\alpha \pm i\beta)$ with $\beta \neq 0$, we compute the ratio $|\alpha_{\mathrm{lead}}|/|\beta_{\mathrm{lead}}|$ and store it as a *bifurcation proximity indicator*. This tells the encoder how close the system is to a Hopf bifurcation, which is useful latent information even when limit cycles are not predicted.


**Remark 6** (Which eigenvalue pair is “leading” for Hopf proximity). *There is a subtle but important distinction between two uses of the word “leading” in the eigenvalue literature that must not be conflated.*

*In the general stability literature (e.g. Scholarpedia’s article on equilibria), the *leading eigenvalue* of a fixed point means the eigenvalue with the *largest real part* across all eigenvalues — real or complex. This is the eigenvalue that governs the slowest decay (or fastest growth) direction and determines whether the fixed point is stable or unstable.*

*For Hopf bifurcation detection, however, we are interested in something different: among all complex conjugate pairs $\alpha_k \pm i\beta_k$ with $\beta_k \neq 0$, we want the pair that is *closest to the imaginary axis*, i.e. the one with the smallest $|\alpha_k|$. This is the pair most likely to cross the imaginary axis as parameters vary, generating a Hopf bifurcation.*

*Concretely, define: $$(\alpha_{\mathrm{lead}}, \beta_{\mathrm{lead}})
= \arg\min_{\{k:\,\beta_k \neq 0\}} |\alpha_k|.
$$ This is *not* the eigenvalue with the largest real part overall (which could be a real eigenvalue), but the complex eigenvalue pair closest to $\pm i\omega$. The Hopf proximity ratio is then: $$\rho_{\mathrm{Hopf}} = \frac{|\alpha_{\mathrm{lead}}|}{|\beta_{\mathrm{lead}}|}.
$$ A small ratio ($\rho_{\mathrm{Hopf}} \ll 1$) indicates the system is near a Hopf bifurcation boundary. If no complex eigenvalue pairs exist ($\beta_k = 0$ for all $k$), the ratio is undefined and no Hopf proximity is recorded.*


#### Saddle-Node Proximity

If $|\det(J(x^*))| < \varepsilon_{\mathrm{SN}}$ at a found fixed point, the system is near a saddle-node bifurcation (a zero eigenvalue). This is flagged as a proximity indicator and triggers additional Newton–Raphson seeds nearby, since another fixed point may be close in state space.

#### Bézout Bound and the Choice of $K_{\mathrm{FP}}$

Under integer Hill coefficients, clearing denominators in the Hill fixed-point equations yields a polynomial system, and Bézout’s theorem gives an upper bound on the number of isolated fixed points in complex projective space with multiplicity. This establishes that the fixed point count is finite — which is the only fact we actually need from Bézout.


**Remark 7** (The Bézout bound is mathematically correct but practically useless for setting $K_{\mathrm{FP}}$). *For a 6-gene SCC where each gene has 3 inputs with Hill coefficient $n=2$, the polynomial degree per equation after clearing denominators is on the order of the product of all Hill factors involved. The Bézout bound gives something of order $6^6 = 46{,}656$ complex projective roots. Setting $K_{\mathrm{FP}}
= 46{,}656$ output slots in the decoder would be computationally absurd.*

*Moreover, the Bézout bound counts roots in complex projective space with multiplicity, which has essentially no relationship to the number of real nonnegative fixed points in a biological GRN under realistic parameters. The biological literature and empirical corpus statistics consistently show that SCCs of size up to $N_{\max} \approx 8$–$10$ under realistic Hill parameters rarely have more than 3–5 stable fixed points and 10–20 total fixed points (including saddles and unstable nodes).*


###### Practical determination of $K_{\mathrm{FP}}$.

$K_{\mathrm{FP}}$ is set as a heuristic hyperparameter based on Corpus A statistics, not from the Bézout bound. The procedure is:

1.  Generate a pilot Corpus A with several thousand SCC realizations spanning the intended range of SCC sizes and topologies.

2.  Plot the empirical histogram of total fixed point counts (stable + unstable + saddle) across all samples.

3.  Set $K_{\mathrm{FP}}$ to the 99th percentile of this distribution plus a small buffer (e.g., $+3$).

Based on the biology and the Li et al. results, we expect $K_{\mathrm{FP}}
\approx 10$–$15$ to be sufficient for SCCs of size up to $N_{\max} = 10$ under realistic Hill parameters. The Bézout argument is retained only to confirm that the fixed point count is finite, justifying a finite decoder.

### Corpus A: Fixed Point Ground Truth Generation

Corpus A is the first and most important corpus. It trains Layer I to predict the Version 1 phase portrait — the complete set of fixed points, their locations, eigenspectra, and stability classes — for any parameterized SCC realization.

#### What a Corpus A Sample Contains

$$\begin{aligned}
\text{Input:}\quad X_A &= \mathcal{G}_i = (G_i, \vartheta_i), \\
\text{Target:}\quad Y_A &= \mathcal{P}_i
= \left\{(x^{*(r)}, \{c_m^{(r)}\}_{m=1}^{n_i}, \sigma^{(r)})\right\}_{r \in \mathcal{I}_{\mathrm{FP}}}.
\end{aligned}$$

The input is the parameterized SCC: its topology (which genes regulate which others, with what sign), and its kinetic parameters $\vartheta_i = (P_i, K_i^{\deg}, T_i, n_i, F_i)$. The target is the complete set of fixed points: where they are in state space, what the characteristic polynomial coefficients of the Jacobian are at each one, and their stability class.

One corpus sample contains all fixed points of one SCC realization. A multistable SCC with three coexisting fixed points contributes one sample with three fixed point records, not three separate samples.

#### Why the Same Topology Can Appear Many Times

The same graph topology $G_i$ can appear many times in Corpus A with different kinetic parameter realizations $\vartheta_i^{(1)}, \vartheta_i^{(2)}, \ldots$ Each distinct parameterization is a separate learning example, because the dynamical repertoire depends on parameters, not just topology. The same toggle switch topology can be monostable under weak cooperativity parameters and bistable under strong cooperativity parameters.

#### SCC Topology Sampling

A critical design decision for Corpus A is how to sample the graph topology $G_i$ of each SCC. Two failure modes must be avoided simultaneously: *over-sampling unrealistically dense graphs* (which produces dynamically unrepresentative SCCs) and *over-constraining topology* (which can bias the dynamical coverage and makes it impossible to study how network structure affects dynamics).

###### Why fully connected graphs are biologically unrealistic.

A fully connected 8-node SCC has $8 \times 7 = 56$ directed edges (ignoring self-loops). Real GRNs are sparse: the average in-degree (number of transcription factors regulating a given gene) is typically 2–5 for most genes. A biologically representative 8-gene regulatory module would have roughly 10–25 edges. Sampling topology uniformly at random from all possible directed graphs on $n$ nodes produces a mean in-degree of $(n-1)/2$, which is far too dense.

###### The directed, asymmetric nature of GRN edges.

GRN edges are directed and biologically asymmetric: in real networks, a small number of master transcription factors (TFs) regulate many target genes (high out-degree), while most genes receive regulation from only a few TFs (low in-degree). In-degree and out-degree distributions are therefore different and should be sampled independently rather than symmetrically.

###### Controlling topology: recommended procedure.

Rather than imposing a rigid topological template, control topology through a small set of tunable parameters:

1.  **SCC size $n_i$.** Sample $n_i$ from the intended size distribution, e.g. $n_i \in \{2, 3, \ldots, N_{\max}\}$ with frequencies reflecting the biological prior (smaller SCCs are more common).

2.  **Edge density.** For an SCC of size $n_i$, sample the total number of directed edges $|E_i|$ uniformly from a biologically motivated range: $$|E_i| \in [n_i,\; c \cdot n_i], \quad c \approx 3,
    $$ with a small fraction of denser samples (up to $5n_i$) for dynamical coverage. The lower bound $|E_i| \geq n_i$ is required by the strong connectivity constraint (each node must have at least one outgoing edge in an SCC).

3.  **In-degree distribution.** For each gene $v$, sample its in-degree $d^-_v$ from a distribution concentrated on small values. A geometric distribution with mean $\mu^- \approx 2$–$3$, truncated at $d^-_{\max} \approx 5$–$6$, is biologically reasonable. This reflects the typical number of TF binding sites in a gene’s promoter.

4.  **Out-degree distribution.** For each gene $v$, sample its out-degree $d^+_v$ from a distribution with a heavier tail than the in-degree distribution, reflecting the existence of hub TFs. A geometric distribution with mean $\mu^+ \approx 2$–$4$ and moderate tail is appropriate. Do not enforce a strict power law.

5.  **Edge signs.** For each edge $u \to v$, assign the regulatory sign (activation $F_{ji} > 1$ or repression $F_{ji} < 1$) according to a fixed probability $p_{\mathrm{rep}} \approx 0.3$–$0.5$. The sign distribution affects the Thomas complexity class of the SCC and should be tuned to ensure adequate coverage of all four classes.

6.  **Self-loops.** Allow but do not require self-loops (TF autoregulation). Include each potential self-loop with probability $p_{\mathrm{self}} \approx 0.2$–$0.3$, consistent with the observed prevalence of TF autoregulation in real GRNs.

7.  **Strong connectivity check.** After sampling the topology, verify that the resulting directed graph is strongly connected (using Tarjan’s algorithm or equivalent). If not, reject and resample. This constraint is non-negotiable: Corpus A samples are by definition SCCs.

###### Stratification by Thomas complexity class.

Random topology sampling with the above parameters will naturally produce a distribution over the four Thomas complexity classes, but this distribution may not provide adequate coverage of all classes. Deliberate stratification is recommended:


| Pos. fb.  | Neg. fb.  | Possible behaviors       | Target fraction |
|:----------|:----------|:-------------------------|:----------------|
| No        | No        | Monostable only          | 20%             |
| Yes       | No        | Multistable only         | 30%             |
| No        | Yes       | Monostable + possible LC | 15%             |
| Yes       | Yes       | Full complexity          | 35%             |


The suggested fractions are approximate; the key point is that the full complexity class (both positive and negative feedback) should be well-represented, since it produces the richest dynamical behavior.

###### What not to constrain.

If one of the research goals is to study how topology affects dynamics — specifically, whether the architecture learns dynamics rather than memorizing topology (Generalization Experiment 1 in Section <a href="#sec:experiments_v1" data-reference-type="ref" data-reference="sec:experiments_v1">17</a>) — then topological diversity is essential. Overly restricting topology to a narrow biological template would:

- Reduce the topological diversity needed for the topology generalization experiment.

- Potentially bias the corpus toward dynamically simpler SCCs, underrepresenting bifurcation-rich topologies.

- Make it harder to evaluate whether the model has learned structure-to-dynamics mappings or is merely interpolating over a narrow topology distribution.

The recommended approach treats topology sampling parameters as tunable hyperparameters that are reported as part of the corpus specification, not as fixed biological ground truth.

#### Kinetic Parameter Sampling

After sampling topology, kinetic parameters $\vartheta_i = (P_i, K_i^{\deg},
T_i, n_i, F_i)$ are sampled independently for each realization. Uniform sampling in linear space is inefficient: most parameter combinations produce monostable dynamics. Stratified sampling in log-space is recommended:


| Parameter                   | Sampling range                                           | Biological motivation                                    |
|:----------------------------|:---------------------------------------------------------|:---------------------------------------------------------|
| $P_i$ (production rate)     | Log-uniform in $[0.01, 100]$                             | Basal expression rates span several orders of magnitude  |
| $K_i^{\deg}$ (degradation)  | Log-uniform in $[0.01, 10]$                              | Protein half-lives vary widely                           |
| $T_{ji}$ (threshold)        | Log-uniform in $[0.1, 10]$                               | Threshold relative to typical expression level           |
| $n_{ji}$ (Hill coefficient) | Uniform in $\{1, 2, 3, 4, 5\}$                           | Cooperative binding; $n \geq 2$ for switch-like behavior |
| $F_{ji}$ (fold-change)      | Log-uniform in $[0.01, 0.5]$ (rep.) or $[2, 100]$ (act.) | Fold-changes of 2–100-fold are biologically common       |


To enrich for multistable and near-bifurcation cases, deliberately oversample parameter combinations with high Hill coefficients ($n_{ji} \geq 3$) and large fold-changes, since these promote cooperative switch-like behavior.

#### Two-Tier Ground Truth Generation

###### Stage 0: Topological pre-classification (Tier 1, free).

Before any numerical computation, inspect the signed graph of $G_i$:

1.  Identify all directed cycles and their sign patterns.

2.  **Check for positive feedback loops** (even number of repressive edges in a cycle). If none: Thomas condition <a href="#eq:thomas1" data-reference-type="eqref" data-reference="eq:thomas1">[eq:thomas1]</a> rules out multistability. At most one stable fixed point. This constrains the decoder output and reduces the multistability search burden.

3.  **Compute the Hopf proximity structure**: identify all directed cycles with an odd number of repressive edges (negative feedback loops). Store the count and lengths of these cycles as free topological descriptors. This information is included as a feature in the corpus sample.

4.  Record the Thomas complexity class of the SCC (one of the four classes in the stratification table above).

###### Stage 1: Fixed point finding (Tier 2).

Apply Newton–Raphson to $f(x) = 0$ from a Latin hypercube sampling of initial guesses inside $\Omega$. For each fixed point $x^{*(r)}$ found:

1.  Record the location $x^{*(r)} \in \mathbb{R}^{n_i}$.

2.  Evaluate the Jacobian $J(x^{*(r)})$ analytically using the known Hill function derivatives.

3.  Compute the characteristic polynomial coefficients $\{c_m^{(r)}\}_{m=1}^{n_i}$ using Newton’s identities: $c_1 = -\operatorname{Tr}(J)$, $c_{n_i} = (-1)^{n_i} \det(J)$, and intermediate coefficients from the traces of powers of $J$.

4.  Compute the index $\operatorname{sgn}(\det(-J^{(r)}))$.

5.  Determine the stability class $\sigma^{(r)}$ from the signs of the real parts of the eigenvalues.

6.  Compute the Hopf proximity ratio <a href="#eq:hopf_ratio" data-reference-type="eqref" data-reference="eq:hopf_ratio">[eq:hopf_ratio]</a> for each fixed point that has complex eigenvalue pairs. Store as a per-fixed-point feature.

7.  Flag near-saddle-node if $|\det(J^{(r)})| < \varepsilon_{\mathrm{SN}}$.

8.  Deduplicate: cluster found roots within a small tolerance $\varepsilon$.

After finding all roots, compute the Brouwer index sum $\sum_r \operatorname{sgn}(\det(-J^{(r)}))$. If this does not equal $1$, additional fixed points likely exist; run additional Newton–Raphson seeds in unexplored regions of $\Omega$ (e.g., near saddle-node proximity flags, or in regions not covered by the initial grid). Record a **completeness flag** indicating whether the index sum was achieved.


**Remark 8** (BVP is not triggered in Version 1). *In Version 1 we do not run BVP continuation for limit cycles. The presence of an unstable spiral is noted in the stability class but does not trigger any additional computation. The Version 1 corpus is therefore entirely algebraic: Newton–Raphson for fixed points, Jacobian evaluation, and Newton’s identity computation. No ODE integration is required.*


#### Corpus A Generation Cost

The computational cost per sample is dominated by Newton–Raphson solves. For an SCC of size $n_i \leq N_{\max} \approx 8$–$10$ and $K_{\mathrm{seeds}}
\approx 500$–$2000$ initial seeds, each Newton solve takes $O(n_i^3)$ per iteration and converges in a small number of iterations for well-conditioned fixed points. The Jacobian evaluation is analytic and exact.

The total cost per sample scales as $O(K_{\mathrm{seeds}} \cdot n_i^3 \cdot
n_{\mathrm{iter}})$, which is very cheap compared to any ODE integration. This is the key computational advantage of Version 1: the entire corpus is generated without any time integration.

### Layer I: Predicting the Fixed Point Phase Portrait

Layer I takes a parameterized SCC and predicts its complete Version 1 phase portrait: all fixed points, their locations, eigenspectra (encoded as characteristic polynomial coefficients), and stability classes.

#### The Encoder

A message-passing graph neural network encodes the parameterized SCC into a compact fixed-size latent representation: $$\boxed{
E_\theta(\mathcal{G}_i) = (z_i^0,\, \{h_{iv}^0\}_{v \in G_i}),
\quad z_i^0 \in \mathbb{R}^d, \quad h_{iv}^0 \in \mathbb{R}^q.
}$$ The graph-level embedding $z_i^0$ captures the global dynamical character of the SCC — how cooperative its feedback is, how many fixed points it is likely to have, whether it is near a bifurcation. The node-level embeddings $\{h_{iv}^0\}$ capture gene-specific information: each gene’s production and degradation rates, and the character of each of its incoming regulatory edges.

Both embeddings are computed by GNN message passing in which genes exchange information with their regulatory neighbors across multiple rounds. After enough rounds, each gene’s embedding reflects not just its own properties but also the properties of genes connected to it through the regulatory network.

**Node features:** $(P_v, K_v^{\deg})$ for each gene $v$.

**Edge features:** $(s_{uv}, T_{uv}, n_{uv}, F_{uv})$ for each edge $u \to v$, where $s_{uv} = +1$ for activation and $-1$ for repression.

**Auxiliary classification head:** A small MLP applied to $z_i^0$ predicts the Thomas complexity class of the SCC (one of four classes) from the graph embedding, supervised by the free topological label computed in Stage 0. This forces the encoder to learn graph-structural features that are predictive of dynamical complexity.

#### The Fixed Point Set Decoder

The decoder maps the graph-level embedding $z_i^0$ and node-level embeddings $\{h_{iv}^0\}$ to an unordered set of fixed point records: $$\boxed{
D_\phi(z_i^0, \{h_{iv}^0\})
= \left\{(p^{(r)}, q^{(r)}, \hat{x}^{*(r)}, \{\hat{c}_m^{(r)}\}_{m=1}^{n_i}, \hat{\sigma}^{(r)})\right\}_{r=1}^{K_{\mathrm{FP}}}.
}
$$

Each slot $r$ produces:

- $p^{(r)} \in [0,1]$: the predicted probability that this slot corresponds to a real fixed point. At inference, slots with $p^{(r)}$ below a threshold are discarded.

- $q^{(r)} \in \mathbb{R}^{d_q}$: the branch token, a latent vector that “represents” this particular fixed point and will be used by Layer II to construct the inter-SCC signal.

- $\hat{x}^{*(r)} \in \mathbb{R}^{n_i}$: the predicted location of the fixed point in state space, produced by gene-specific readout heads: $$\hat{x}_v^{*(r)} = \mathrm{FPReadOut}_\phi(h_{iv}^0, q^{(r)}) \in \mathbb{R},
  \quad \forall v \in G_i.$$

- $\{\hat{c}_m^{(r)}\}_{m=1}^{n_i}$: the predicted characteristic polynomial coefficients of $J(\hat{x}^{*(r)})$, from which the eigenspectrum can be recovered. These are predicted by an MLP applied to $z_i^0$ and $q^{(r)}$.

- $\hat{\sigma}^{(r)}$: the predicted stability class, derived deterministically from the predicted characteristic polynomial coefficients. Never independently predicted, preventing inconsistency.


**Remark 9** (Why characteristic polynomial coefficients, not eigenvalues). *Eigenvalues $\{\lambda_k\}$ are an unordered set of complex numbers forming conjugate pairs. Predicting them directly creates two problems: the supervised loss $\sum_k |\hat{\lambda}_k - \lambda_k|^2$ requires matching predicted eigenvalues to ground truth eigenvalues, which is ambiguous; and the model could predict a set of eigenvalues that is inconsistent with any real matrix.*

*Characteristic polynomial coefficients $\{c_m\}$ avoid both problems: they are elementary symmetric polynomials of the eigenvalues, automatically permutation-invariant, and always consistent with a real matrix. The loss $\sum_m (\hat{c}_m - c_m)^2$ is well-defined without any matching.*


#### Layer I Training Loss

$$\boxed{
\mathcal{L}_A
= \mathcal{L}_{\mathrm{set}}
+ \lambda_{\mathrm{phys}} \mathcal{L}_{\mathrm{physics}}
+ \lambda_{\mathrm{eig}} \mathcal{L}_{\mathrm{eig}}
+ \lambda_{\mathrm{Th}} \mathcal{L}_{\mathrm{Thomas}}.
}
$$

###### Set loss $\mathcal{L}_{\mathrm{set}}$.

The phase portrait is an unordered set of fixed points. The loss uses the Hungarian algorithm to find the optimal assignment $\pi^*$ between predicted slots and ground truth fixed points (minimizing a suitable matching cost), then measures discrepancy for matched pairs: $$\begin{aligned}
\mathcal{L}_{\mathrm{set}}
&= \mathbb{E}_{\mathcal{G}_i}\!\left[
\min_\pi \sum_r \Bigl(
\mathcal{L}_{\mathrm{exist}}(p^{(r)}, \mathbb{1}[\pi(r) \text{ exists}])
+ \mathbb{1}[\pi(r) \text{ exists}] \cdot \mathcal{L}_{\mathrm{desc}}(r, \pi(r))
\Bigr)
\right], \\
\mathcal{L}_{\mathrm{desc}}(r, r')
&= \|\hat{x}^{*(r)} - x^{*(r')}\|^2
+ \sum_{m=1}^{n_i} (\hat{c}_m^{(r)} - c_m^{(r')})^2.
\end{aligned}$$ The existence loss $\mathcal{L}_{\mathrm{exist}}$ is binary cross-entropy. Unmatched predicted slots (false positive predictions) contribute only the existence loss; unmatched ground truth fixed points (missed predictions) also contribute only the existence loss.

###### Vector field residual $\mathcal{L}_{\mathrm{physics}}$.

At any true fixed point, $f(x^*) = 0$ exactly. For each predicted slot matched to a ground truth fixed point, we can evaluate this residual analytically using the known Hill functions: $$\boxed{
\mathcal{L}_{\mathrm{physics}}
= \sum_{r:\, \pi^*(r) \text{ exists}}
\sum_{i \in G_j}
\left(
P_i \prod_{j \to i} H_{ji}(\hat{x}_j^{*(r)}) - K_i^{\deg} \hat{x}_i^{*(r)}
\right)^2.
}
$$ This loss requires no additional labeled data: just the predicted fixed point location and the known Hill model. It penalizes the model for predicting “fixed points” at locations where the ODE flow is nonzero, preventing hallucinated fixed points. It is applied only to matched records, not to all $K_{\mathrm{FP}}$ slots, to avoid incentivizing the model to lower $p^{(r)}$ to escape the physics penalty.

###### Eigenspectrum consistency $\mathcal{L}_{\mathrm{eig}}$.

The predicted characteristic polynomial coefficients $\{\hat{c}_m^{(r)}\}$ must be consistent with the Jacobian evaluated analytically at the predicted fixed point $\hat{x}^{*(r)}$: $$\boxed{
\mathcal{L}_{\mathrm{eig}}
= \sum_{r:\, \pi^*(r) \text{ exists}}
\sum_{m=1}^{n_i}
\left(\hat{c}_m^{(r)} - c_m(J(\hat{x}^{*(r)}))\right)^2.
}
$$ This provides $n_i$ free consistency constraints per matched fixed point: the trace constraint ($m=1$: sum of predicted eigenvalues equals trace of $J$), the determinant constraint ($m=n_i$: product of predicted eigenvalues equals determinant of $J$), and all intermediate symmetric function constraints. The Jacobian $J(\hat{x}^{*(r)})$ is computable analytically from the Hill functions at the predicted location, making this a free loss requiring no additional data.

###### Thomas exclusion loss $\mathcal{L}_{\mathrm{Thomas}}$.

When no positive feedback loop exists in $G_i$, predicting more than one stable fixed point violates the Thomas necessary condition <a href="#eq:thomas1" data-reference-type="eqref" data-reference="eq:thomas1">[eq:thomas1]</a>: $$\mathcal{L}_{\mathrm{Thomas}}
= \mathbb{1}[\text{no positive feedback in } G_i]
\cdot \sum_{r \geq 2} p^{(r)} \cdot \mathbb{1}[\hat{\sigma}^{(r)} = \text{stable}].
$$ This penalizes the model for predicting multistability when it is provably impossible. It is asymmetric by design: it does not reward predicting multistability when positive feedback is present, because presence of positive feedback does not guarantee multistability.

### Constant Inter-SCC Signal Construction

In Version 1, every upstream SCC is at a stable fixed point. The signal it sends to a downstream SCC is therefore a constant scalar.

#### The Physical Basis

Let source gene $a \in G_i$ regulate target gene $b \in G_j$ via a directed edge with Hill parameters $\xi_{ab} = (T_{ab}, n_{ab}, F_{ab})$. When the source SCC is at stable fixed point $r$ with predicted location $\hat{x}^{*(r)}$, gene $a$ has expression level $\hat{x}_{i,a}^{*(r)}$. The regulatory signal sent to gene $b$ is obtained by passing this expression level through the Hill function with the edge-specific parameters: $$\boxed{
s_r = H_{ab}(\hat{x}_{i,a}^{*(r)};\, T_{ab}, n_{ab}, F_{ab}) \in \mathbb{R}.
}
$$ This is a constant scalar: the regulatory input that gene $a$’s expression level at equilibrium applies to gene $b$’s production term.

#### Why the Hill Transformation is Analytic

The signal $s_r$ is computed analytically, not by a neural network. This is important for several reasons:

1.  The Hill function is a known physical law describing how transcription factor binding modulates gene production. It should not be relearned.

2.  The analytic transformation is differentiable with respect to $\hat{x}_{i,a}^{*(r)}$, so gradients from the downstream composition loss flow back through the Hill function into the source SCC’s decoder.

3.  The edge parameters $\xi_{ab}$ are known inputs (they are part of the GRN specification), not latent variables.

#### The Readout Module is Eliminated

An earlier version of this architecture included a learned readout module $R_\rho$ that mapped the branch token $q^{(r)}$ and node embedding $h_{ia}^0$ to the source gene’s equilibrium value. This module is eliminated: the fixed point location $\hat{x}_{i,a}^{*(r)}$ is directly predicted by the gene-specific readout head of the decoder (as part of $\hat{x}^{*(r)}$), and the Hill transformation is applied analytically. There is no need for a separate learned readout.

#### Constant Forcing and Autonomous Receiving SCC

When $s_r$ is a constant, the receiving SCC’s ODE at gene $b$ becomes: $$\dot{x}_b = P_b \underbrace{\prod_{c \to b,\, c \in G_j} H_{cb}(x_c)}_{R_b(\bm{x}_{G_j})} \cdot s_r - K_b^{\deg} x_b,
$$ where $R_b(\bm{x}_{G_j})$ is the product of internal Hill factors from regulators of gene $b$ within $G_j$. Because $s_r$ is constant, the receiving SCC remains an *autonomous* ODE system. Its phase portrait — fixed points, their stability, their eigenspectra — has the same mathematical character as the autonomous phase portrait of any SCC. Layer III therefore predicts a Version 1 phase portrait for the conditioned receiving SCC, using the same decoder $D_\phi$ as Layer I.

#### Multiple External Edges

If gene $b \in G_j$ receives signals from multiple upstream SCCs (via edges from source genes $a_1, a_2, \ldots$ in different source SCCs), the signals combine multiplicatively, consistent with the Hill product structure of <a href="#eq:grn" data-reference-type="eqref" data-reference="eq:grn">[eq:grn]</a>: $$\dot{x}_b = P_b R_b(\bm{x}_{G_j}) \cdot \prod_k s_r^{(k)} - K_b^{\deg} x_b,
$$ where $s_r^{(k)} = H_{a_k b}(\hat{x}_{i_k, a_k}^{*(r_k)}; \xi_{a_k b})$ is the scalar signal from source $k$. The product of scalars is a scalar, so the receiving SCC remains autonomous under multi-port constant forcing.

### Layer II: Computing Interaction Messages

Layer II combines the typed signal $s_r$ with information about the receiving SCC to produce a branch-specific interaction message. In Version 1, all signals are constant scalars, so the message construction is clean and uniform.

$$\boxed{
m_{(i,a)\to(j,b)}^{(r)}
= \Phi_\psi(s_r,\, z_j^0,\, h_{jb}^0,\, \xi_{ab}) \in \mathbb{R}^m.
}
$$

The inputs to $\Phi_\psi$ are:

- $s_r \in \mathbb{R}$: the constant signal from source branch $r$, computed analytically via the Hill function.

- $z_j^0 \in \mathbb{R}^d$: the graph-level embedding of the receiving SCC $G_j$ from Layer I, encoding its autonomous dynamical character.

- $h_{jb}^0 \in \mathbb{R}^q$: the node embedding of the specific target gene $b$ where the signal enters, encoding gene $b$’s local regulatory context.

- $\xi_{ab} = (T_{ab}, n_{ab}, F_{ab})$: the edge Hill parameters, passed explicitly so $\Phi_\psi$ can learn how the character of the edge modulates the signal’s effect on the target.

$\Phi_\psi$ is implemented as a standard MLP acting on the concatenated vector $[s_r \,\|\, z_j^0 \,\|\, h_{jb}^0 \,\|\, \xi_{ab}]$. The same $\Phi_\psi$ is shared across all edges, all source branch indices, and all SCC size pairings. The message $m^{(r)}$ is a latent vector that encodes “what this particular signal level, entering at this particular gene in this particular SCC, implies for the conditioned dynamics.”

The complete set of messages from all stable source branches of $G_i$ to target gene $b \in G_j$ is: $$\mathcal{M}_{i \to j} = \{m_{(i,a)\to(j,b)}^{(r)}\}_{r \in \mathcal{I}_{\mathrm{stable},i}},$$ one message per stable fixed point branch of the source SCC.

### Corpus B: Single-SCC Constant Forced Response

Corpus B trains the response operator: given a receiving SCC $G_j$ and a constant forcing signal at one or more of its genes, predict the conditioned phase portrait. This is the essential corpus for teaching Layer II and Layer III how SCCs respond to upstream regulatory inputs.

#### The Decoupling Principle

The receiving SCC does not need to know which upstream SCC is sending the signal. It only experiences the signal $s_r$ arriving at gene $b$. Once $s_r$ is computed analytically from the source SCC’s predicted fixed point via the Hill function, the source SCC’s identity is irrelevant. Corpus B is therefore a single-SCC dataset: no source SCC appears anywhere in the corpus.

This decoupling is crucial: it means the response operator learned from Corpus B can be applied at inference to any upstream source, as long as the signal $s_r$ is correctly computed. Composition is zero-shot with respect to the source SCC identity.

#### Corpus B Sample Specification

$$\boxed{
X_B = \left(\mathcal{G}_j,\; \{(b_k, u_k, \xi_k)\}_{k=1}^L\right),
\quad L \in \{1, \ldots, L_{\max}\},
}
$$ where $b_k \in G_j$ is a target gene, $u_k \in (\min(1,F_k), \max(1,F_k))$ is a synthetic constant forcing level, and $\xi_k = (T_k, n_k, F_k)$ are synthetic Hill edge parameters. Multiple ports ($L > 1$) handle the case where $G_j$ receives signals from multiple upstream SCCs simultaneously.

The synthetic forcing levels $u_k$ are sampled to cover the realistic range of Hill function outputs: uniformly in $(\min(1, F_{ji}), \max(1, F_{ji}))$ for representative Hill parameters. The sampling should also deliberately oversample near bifurcation boundaries (near values of $u_k$ where the number or stability of fixed points changes), since these cases are important for generalization but rare under uniform sampling.

#### Ground Truth Generation for Corpus B

For each Corpus B sample, the receiving SCC is solved under constant forcing. The forced ODE for each target gene $b_k$ is: $$\dot{x}_{b_k} = P_{b_k} R_{b_k}(\bm{x}_{G_j}) \cdot \prod_{k':\, b_{k'} = b_k} u_{k'} - K_{b_k}^{\deg} x_{b_k},
$$ where $R_{b_k}(\bm{x}_{G_j}) = \prod_{c \to b_k,\, c \in G_j} H_{c,b_k}(x_c)$ is the internal Hill product for gene $b_k$ within $G_j$. Multiple external ports targeting the same gene combine multiplicatively.

Since the forcing is constant, this is an autonomous ODE. The ground truth conditioned phase portrait is: $$Y_B = \mathcal{P}_j^{\mathrm{cond}}(\{u_k\}, \{b_k\})
= \left\{(x^{*(r)}, \{c_m^{(r)}\}, \sigma^{(r)})\right\}_{r \in \mathcal{I}_{\mathrm{FP}}^{\mathrm{cond}}},$$ generated by the same two-tier pipeline as Corpus A (Newton–Raphson for fixed points, Jacobian evaluation, characteristic polynomial computation, Brouwer check), applied to the forced system <a href="#eq:corpusB_ode" data-reference-type="eqref" data-reference="eq:corpusB_ode">[eq:corpusB_ode]</a>.


**Remark 10** (Why the conditioned phase portrait can differ qualitatively from the autonomous one). *The constant forcing $u_k$ effectively changes the production rate of gene $b_k$. This can push the SCC across bifurcation boundaries:*

- *A forcing level that increases production may create a new stable fixed point (bifurcation from monostability to bistability).*

- *A forcing level that decreases production may eliminate a stable fixed point (saddle-node bifurcation, leaving a single stable fixed point).*

- *The stability of existing fixed points may change (a stable spiral may become a stable node, or vice versa, as the eigenspectrum shifts).*

*The model must learn to predict these qualitative changes, not just quantitative shifts in fixed point locations. This is why stratification near bifurcation boundaries is essential.*


#### Required Corpus B Stratification

Naive uniform sampling of $u_k$ will heavily oversample the generic case (small quantitative shift in fixed point locations) and undersample the important cases (qualitative changes in the phase portrait). Deliberate stratification is required:


| Conditioning outcome                                 | Stratification strategy                                                 |
|:-----------------------------------------------------|:------------------------------------------------------------------------|
| Quantitative shift in fixed point locations          | Common; standard uniform sampling                                       |
| Change in stability class of a fixed point           | Enrich near values of $u_k$ where $\alpha_{\mathrm{lead}}$ changes sign |
| Saddle-node bifurcation (gain or lose a fixed point) | Enrich near values of $u_k$ where $|\det(J)| \approx 0$                 |
| Number of fixed points changes                       | Use parameter continuation to trace bifurcation curves                  |


#### Corpus B Forward Map and Loss

The neural forward pass for Corpus B sample $(X_B, Y_B)$ is: $$\begin{aligned}
m_k &= \Phi_\psi(u_k,\, z_j^0,\, h_{jb_k}^0,\, \xi_k)
\quad \forall k, \\
M_j &= A_\eta(\{m_k\}_{k=1}^L), \\
(z_j^{\mathrm{cond}}, \{h_{jv}^{\mathrm{cond}}\})
&= U_\omega(G_j,\, z_j^0,\, \{h_{jv}^0\},\, M_j,\, D_\phi(z_j^0, \{h_{jv}^0\})),
\\
\widehat{\mathcal{P}}_j^{\mathrm{cond}}
&= D_\phi(z_j^{\mathrm{cond}}, \{h_{jv}^{\mathrm{cond}}\}). 
\end{aligned}$$

Note that $A_\eta$ (the message aggregator) is trained in Stage 2 on Corpus B, not only in Stage 3 on Corpus C, because multi-port forcing ($L > 1$) already appears in Corpus B. This ensures $A_\eta$ and $U_\omega$ learn multi-gene signal interaction before encountering full DAGs.

The Corpus B loss uses the same physics-informed terms as Corpus A, applied to the conditioned predicted phase portrait: $$\boxed{
\mathcal{L}_B
= \mathcal{L}_{\mathrm{set}}(\widehat{\mathcal{P}}_j^{\mathrm{cond}}, \mathcal{P}_j^{\mathrm{cond}})
+ \lambda_{\mathrm{phys}} \mathcal{L}_{\mathrm{physics}}
+ \lambda_{\mathrm{eig}} \mathcal{L}_{\mathrm{eig}}
+ \lambda_{\mathrm{Th}} \mathcal{L}_{\mathrm{Thomas}}.
}
$$ The physics losses here enforce that the predicted conditioned fixed points satisfy the forced ODE <a href="#eq:corpusB_ode" data-reference-type="eqref" data-reference="eq:corpusB_ode">[eq:corpusB_ode]</a>, not the autonomous ODE. Specifically, $\mathcal{L}_{\mathrm{physics}}$ evaluates $f_{\mathrm{forced}}(x^*)$ where $f_{\mathrm{forced}}$ includes the constant forcing term.

### Layer III: Predicting the Conditioned Phase Portrait

Layer III takes the aggregated message from all incoming inter-SCC edges and produces a conditioned representation of the receiving SCC. In Version 1, all signals are constant and the conditioned system is autonomous, so Layer III predicts a standard fixed point phase portrait.

#### The Update Step

$$\boxed{
(z_j^{\mathrm{cond}}, \{h_{jv}^{\mathrm{cond}}\})
= U_\omega\!\left(G_j,\, z_j^0,\, \{h_{jv}^0\},\, M_j,\, D_\phi(z_j^0, \{h_{jv}^0\})\right).
}
$$

$U_\omega$ is a GNN operating on the topology of $G_j$ (the internal regulatory graph of the receiving SCC). The aggregated message $M_j$ is injected as a conditioning signal at the entry gene(s), for example via FiLM-style modulation of the GNN’s hidden states. All node representations are updated — not just the entry gene $b$ — because the signal enters at $b$ but propagates through $G_j$’s internal feedback structure to affect all genes.

**Bifurcation proximity conditioning**: the autonomous phase portrait $D_\phi(z_j^0, \{h_{jv}^0\})$ is passed explicitly to $U_\omega$. This gives $U_\omega$ direct access to the autonomous dynamical character of the receiving SCC — in particular, how close it is to saddle-node or Hopf bifurcation boundaries. A receiving SCC near a saddle-node bifurcation will lose or gain a fixed point under a smaller signal perturbation than one far from bifurcation. Conditioning on this information makes qualitative transition prediction learnable.

#### Decoding the Conditioned Phase Portrait

The same decoder $D_\phi$ used in Layer I is applied to the conditioned embedding to predict the conditioned phase portrait: $$\boxed{
\widehat{\mathcal{P}}_j^{\mathrm{cond}}
= D_\phi(z_j^{\mathrm{cond}}, \{h_{jv}^{\mathrm{cond}}\}).
}
$$

Sharing $D_\phi$ between Layer I and Layer III is valid in Version 1 because both output the same type of object: an autonomous phase portrait consisting of fixed points with characteristic polynomial coefficients. The shared decoder is trained on both Corpus A (autonomous) and Corpus B (constant forced), so it learns to predict fixed point phase portraits in both settings.

### Multiple Incoming Edges and Message Aggregation

When the receiving SCC $G_j$ has multiple parent SCCs, each potentially at different stable fixed point branches, the messages from all incoming edges must be aggregated: $$M_j^{(\beta)} = A_\eta(\{m_e^{(\beta)} : e \to G_j\}),
$$ where $\beta$ is a compatible branch assignment specifying one stable fixed point branch per parent SCC.

$A_\eta$ is a permutation-invariant aggregator (e.g., DeepSets or attention) since the order of incoming edges is arbitrary. All messages $m_e^{(\beta)} \in \mathbb{R}^m$ regardless of which parent they come from, so no type-aware modification is needed.

$$(z_j^{\mathrm{cond}|\beta}, \{h_{jv}^{\mathrm{cond}|\beta}\})
= U_\omega(G_j, z_j^0, \{h_{jv}^0\}, M_j^{(\beta)}, D_\phi(z_j^0, \{h_{jv}^0\})).$$

### Corpus C: Complete SCC-DAG Composition

Corpus C teaches the architecture to correctly chain composition operators across complete SCC-DAGs, validating that recursive reuse of $\Phi_\psi$, $A_\eta$, $U_\omega$, and $D_\phi$ remains accurate at depth.

#### Why Corpus C Is Needed

Corpus B teaches one-step composition: one source SCC sends a signal to one target SCC. But in a real GRN the DAG may be many levels deep: $G_1 \to G_2 \to G_3 \to G_4 \to \ldots$ When Layer III’s output for $G_2$ feeds into Layer II’s input for $G_3$, errors can compound. Corpus C teaches the architecture to remain accurate under recursive chaining.

#### Sample Specification

$$\begin{aligned}
X_C &= \bigl(\mathcal{G}_{\mathrm{SCC}},\, \{\mathcal{G}_i\}_{i=1}^M,\,
\{(i, a_e, j, b_e, T_e, n_e, F_e)\}_{e \in \mathcal{E}}\bigr), \\
Y_C &= \{\text{conditioned fixed point phase portrait at every SCC in the DAG}\}.
\end{aligned}$$

Ground truth is generated in topological order: for each SCC $G_j$, compute the constant forcing signals from all upstream SCCs (using the previously computed conditioned fixed points of those SCCs), solve the forced system via Newton–Raphson, and store the resulting conditioned phase portrait.

#### Branch Compatibility at Reconvergent Nodes

Consider the diamond structure: $G_1 \to G_2$, $G_1 \to G_3$, $G_2 \to G_4$, $G_3 \to G_4$. Source branch $A_r$ of $G_1$ determines:

- the forcing signal $s_{G_2}^{(r)}$ received by $G_2$,

- the forcing signal $s_{G_3}^{(r)}$ received by $G_3$.

These two signals are co-determined by the same ancestor branch $A_r$: they cannot be mixed across different ancestor branches.

Under forcing $s_{G_2}^{(r)}$, $G_2$ may have multiple stable fixed points $\{B_i^{(r)}\}$. Under forcing $s_{G_3}^{(r)}$, $G_3$ may have multiple stable fixed points $\{C_j^{(r)}\}$. All combinations $(B_i^{(r)}, C_j^{(r)})$ for the same ancestor $r$ are potentially physically realizable (depending on the initial conditions of $G_2$ and $G_3$). Cross-ancestry combinations $(B_i^{(r)}, C_j^{(r')})$ for $r \neq r'$ are not realizable because they would require $G_1$ to be in two different stable states simultaneously.

The architecture tracks ancestor branch identity to enforce this compatibility constraint.

#### Required Stratification


| Dimension                    | Coverage                                                    |
|:-----------------------------|:------------------------------------------------------------|
| DAG structure                | Chains, trees, diamonds, multi-path reconvergent            |
| DAG depth                    | Depth 2 through 6+                                          |
| Multistability along lineage | Monostable throughout; bistability at one or more levels    |
| Bifurcation events           | No bifurcations; single bifurcation at one level; cascading |
| Fan-in at reconvergent nodes | 2 parents; 3+ parents                                       |


### Recursive Inference on a Large GRN

Given a new large GRN at inference time:

1.  **Decompose.** Identify SCCs using Tarjan’s algorithm or equivalent. Build the condensation DAG.

2.  **Classify.** For each SCC, apply Tier 1 topological pre-classification: check for positive feedback loops (Thomas condition).

3.  **Encode.** For each SCC independently: $E_\theta(\mathcal{G}_i) = (z_i^0, \{h_{iv}^0\})$. The encoder never sees more than $N_{\max}$ nodes at a time.

4.  **Decode autonomous portraits.** For each SCC: $D_\phi(z_i^0, \{h_{iv}^0\}) = \widehat{\mathcal{P}}_i^{\mathrm{auto}}$. This predicts the fixed points of each SCC in isolation.

5.  **Process in topological order.** For each receiving SCC $G_j$, from most upstream to most downstream:

    1.  For each parent $G_i$ with edge $a \to b$ and parameters $\xi_{ab}$: read the predicted stable fixed point branches from $\widehat{\mathcal{P}}_i^{\mathrm{cond}}$ (or $\widehat{\mathcal{P}}_i^{\mathrm{auto}}$ if $G_i$ has no parents); compute the scalar signal $s_r = H_{ab}(\hat{x}_{i,a}^{*(r)}; \xi_{ab})$; compute the message $m^{(r)} = \Phi_\psi(s_r, z_j^0, h_{jb}^0, \xi_{ab})$.

    2.  Determine compatible branch assignments $\beta$: enumerate all combinations of stable source branches from all parents, respecting ancestor compatibility at reconvergent nodes.

    3.  Aggregate: $M_j^{(\beta)} = A_\eta(\{m_e^{(\beta)}\})$.

    4.  Update: $(z_j^{\mathrm{cond}|\beta}, \{h_{jv}^{\mathrm{cond}|\beta}\})
        = U_\omega(G_j, z_j^0, \{h_{jv}^0\}, M_j^{(\beta)}, D_\phi(z_j^0, \{h_{jv}^0\}))$.

    5.  Decode: $\widehat{\mathcal{P}}_j^{(\beta)}
        = D_\phi(z_j^{\mathrm{cond}|\beta}, \{h_{jv}^{\mathrm{cond}|\beta}\})$.

    6.  Store $\widehat{\mathcal{P}}_j^{(\beta)}$ and conditioned embeddings for use by downstream SCCs.

6.  **Report.** The full GRN’s predicted dynamical repertoire is the collection of conditioned phase portraits at every SCC in the DAG, organized by branch lineage.

### Complete Version 1 Architecture

$$\begin{aligned}
E_\theta:&\quad \mathcal{G}_i \mapsto (z_i^0, \{h_{iv}^0\}), \\
D_\phi:&\quad (z, \{h_v\}) \mapsto \{(p^{(r)}, q^{(r)}, \hat{x}^{*(r)}, \{\hat{c}_m^{(r)}\})\}, \\
\Phi_\psi:&\quad (s_r, z_j^0, h_{jb}^0, \xi_{ab}) \mapsto m^{(r)}, \\
A_\eta:&\quad \{m_e^{(\beta)}\} \mapsto M_j^{(\beta)}, \\
U_\omega:&\quad (G_j, z_j^0, \{h_{jv}^0\}, M_j^{(\beta)}, D_\phi(z_j^0, \{h_{jv}^0\})) \mapsto (z_j^{\mathrm{cond}}, \{h_{jv}^{\mathrm{cond}}\}).
\end{aligned}$$


| **Layer** | **Maps**                    | **Role**                                                                                                                                                                                                                                                                                                                                                                     |
|:----------|:----------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| I         | $E_\theta, D_\phi$          | Encode a parameterized autonomous SCC into fixed-size graph- and node-level latent representations. Decode the autonomous fixed point phase portrait: location, characteristic polynomial coefficients (permutation-invariant eigenspectrum), stability class. Physics losses enforce self-consistency on matched predictions. Brouwer constraint used as evaluation metric. |
| II        | $\Phi_\psi$ + analytic Hill | Compute the constant scalar inter-SCC signal by applying the Hill function analytically to the predicted source fixed point value. Combine with target SCC context and edge parameters to produce a branch-specific interaction message.                                                                                                                                     |
| III       | $A_\eta, U_\omega, D_\phi$  | Aggregate messages from all incoming parent branches. Propagate through the receiving SCC graph with bifurcation proximity conditioning. Decode the conditioned fixed point phase portrait, which may differ qualitatively from the autonomous one if the signal crosses a bifurcation boundary.                                                                             |


Full pipeline for one edge and one source branch: $$\boxed{
D_\phi(z_i^0, \{h_{iv}^0\})\big|_r
\xrightarrow{H_{ab}(\cdot;\,\xi_{ab})}
s_r \in \mathbb{R}
\xrightarrow{\Phi_\psi(\cdot,\,z_j^0,\,h_{jb}^0,\,\xi_{ab})}
m^{(r)}
\xrightarrow{U_\omega(\cdot,\,D_\phi(z_j^0,\,\{h_{jv}^0\}))}
z_j^{\mathrm{cond}}
\xrightarrow{D_\phi}
\widehat{\mathcal{P}}_j^{\mathrm{cond}}.
}$$

### Training Program (Version 1)

1.  **Autonomous fixed point learning (Corpus A).**

    Train $E_\theta$ and $D_\phi$ jointly on Corpus A using loss <a href="#eq:lossA" data-reference-type="eqref" data-reference="eq:lossA">[eq:lossA]</a>. The physics losses $\mathcal{L}_{\mathrm{physics}}$ and $\mathcal{L}_{\mathrm{eig}}$ are active throughout and provide free training signal that does not require additional labeled data. The Thomas exclusion loss $\mathcal{L}_{\mathrm{Thomas}}$ enforces topological consistency.

    The auxiliary Thomas complexity-class classification head is trained jointly.

    At the end of Stage 1, the encoder and decoder can predict the complete fixed point phase portrait of any SCC realization, with predictions that are self-consistent with the governing equations.

2.  **Constant forced response learning (Corpus B).**

    Train $\Phi_\psi$, $A_\eta$, and $U_\omega$ on Corpus B using loss <a href="#eq:lossB" data-reference-type="eqref" data-reference="eq:lossB">[eq:lossB]</a>. The autonomous encoder $E_\theta$ is initially frozen; after convergence, it may be fine-tuned at a reduced learning rate so that gradients from the downstream composition loss can improve the source SCC representation.

    $D_\phi$ is fine-tuned simultaneously so that it learns to decode conditioned phase portraits (under constant forcing) in addition to autonomous ones. The physics losses in $\mathcal{L}_B$ use the forced ODE <a href="#eq:corpusB_ode" data-reference-type="eqref" data-reference="eq:corpusB_ode">[eq:corpusB_ode]</a>, not the autonomous ODE.

    At the end of Stage 2, the architecture can predict the conditioned fixed point phase portrait of any receiving SCC under any constant scalar forcing, for any target gene, without knowing the identity of the source SCC.

3.  **Recursive DAG composition (Corpus C).**

    Train the full recursive pipeline on Corpus C. The branch token $q^{(r)}$ from the source SCC’s decoder output feeds into $\Phi_\psi$ via the signal construction, so gradients flow through the full chain.

    Begin with teacher-forced branch information (the true upstream conditioned phase portrait is provided at each step) and gradually transition to free-running inference (the model’s own upstream predictions are used). This curriculum avoids gradient explosion from compounding errors early in training.

    Stratify mini-batches to ensure adequate coverage of bifurcating cases, reconvergent structures, and deep DAGs.

### Evaluation Metrics (Version 1)

###### Fixed point count accuracy.

Exact-match accuracy of the predicted number of fixed points (stable nodes, unstable nodes, saddles) versus ground truth.

###### Fixed point location error.

After optimal set matching, mean squared error $\|\hat{x}^{*(r)} - x^{*(r)}\|^2$ per matched fixed point.

###### Characteristic polynomial consistency.

MSE between predicted $\hat{c}_m^{(r)}$ and ground truth $c_m^{(r)}$ after matching; also MSE between $\hat{c}_m^{(r)}$ and $c_m(J(\hat{x}^{*(r)}))$ (the free consistency check from $\mathcal{L}_{\mathrm{eig}}$).

###### Stability class accuracy.

Per-fixed-point classification accuracy across stability classes (stable node, stable spiral, unstable node, unstable spiral, saddle).

###### Coexistence structure accuracy.

Fraction of SCC realizations where the predicted coexistence structure (monostable / bistable / tristable / ...) exactly matches ground truth.

###### Vector field residual.

Mean $\|f(\hat{x}^{*(r)})\|^2$ over matched predicted fixed points.

###### Brouwer index sum violation rate.

Fraction of predicted fixed point sets satisfying $\sum_r p^{(r)} \operatorname{sgn}(\det(-\hat{J}^{(r)})) \approx 1$.

###### Thomas violation rate.

Fraction of predictions that predict multistability for SCCs without positive feedback loops.

###### Conditioned portrait accuracy.

Same metrics as above, applied to the conditioned phase portrait $\widehat{\mathcal{P}}_j^{\mathrm{cond}}$ from Corpus B and C evaluations.

###### Zero-shot composition accuracy.

For $(G_i, G_j)$ source-target SCC pairings never seen together during Corpus B training: accuracy of the conditioned phase portrait prediction.

###### Depth-error curves.

All metrics plotted as a function of condensation-DAG depth under free-running Corpus C inference.

###### Branch compatibility rate.

At reconvergent DAG nodes: fraction of predicted branch assignments that are physically compatible with the common ancestry constraint.

### Generalization Experiments (Version 1)


| **Experiment**            | **Training**                                      | **Testing**                            | **Question**                                                                    |
|:--------------------------|:--------------------------------------------------|:---------------------------------------|:--------------------------------------------------------------------------------|
| Topology                  | Subset of topologies                              | Unseen topologies of seen sizes        | Does Layer I learn dynamics, not memorize topology?                             |
| Parameter transfer        | $\vartheta \in \Omega_{\mathrm{train}}$           | $\vartheta \in \Omega_{\mathrm{test}}$ | Does Layer I generalize across kinetic parameter realizations?                  |
| Physics loss ablation     | With vs. without $\mathcal{L}_{\mathrm{physics}}$ | Same test set                          | How many hallucinated FPs does the residual loss eliminate?                     |
| Eigenspectrum consistency | With vs. without $\mathcal{L}_{\mathrm{eig}}$     | Same test set                          | Does the characteristic polynomial loss improve stability accuracy?             |
| Bifurcation detection     | Few bifurcating Corpus B samples                  | Enriched bifurcating test set          | Can the model predict qualitative phase portrait changes from constant forcing? |
| Zero-shot composition     | Corpus B synthetic forcings                       | Real $(G_i, G_j)$ pairings             | Does the response operator transfer to real upstream signals?                   |
| Fan-in                    | Small message sets                                | Larger parent sets                     | Does $A_\eta$ generalize to more incoming edges than seen in training?          |
| DAG depth                 | Shallow (2–3 levels)                              | Deep (5+ levels)                       | How fast does error accumulate with recursive depth?                            |
| Multistability at depth   | Monostable DAGs                                   | Multistable SCCs at multiple levels    | Does the architecture correctly propagate multistability information?           |
| Reconvergence             | Chain and tree DAGs                               | Diamond and multi-path DAGs            | Is branch compatibility preserved at reconvergent nodes?                        |
| Topology density          | Sparse training graphs                            | Denser test graphs                     | Does the model degrade gracefully when test SCCs are denser than training SCCs? |
| Thomas class transfer     | Balanced class training                           | Held-out Thomas class                  | Does the architecture generalize across dynamical complexity classes?           |


## Part II — Oscillatory Extension (Version 2)

### When and Why to Extend to Version 2

Version 1 covers the dominant dynamical behaviors in real biological GRNs: monostability, multistability, and the fixed point phase portraits that underlie cell fate decisions. The architecture should be built and validated on Version 1 first.

Version 2 extends the architecture to handle limit cycles and periodic inter-SCC signals. This extension is motivated by biologically important network classes — circadian clocks, cell cycle engines, metabolic oscillators — where sustained oscillation is the primary functional output. However, Version 2 adds substantial complexity and should only be pursued after Version 1 has been empirically validated.

The empirical trigger for proceeding to Version 2: after Version 1 is trained and evaluated, inspect the completeness flags in Corpus A. If the fraction of SCC realizations that triggered BVP continuation attempts (i.e., those with near-Hopf signatures) is substantial and the Version 1 model fails systematically on these cases, proceed to Version 2.

### What Changes in Version 2

Version 2 makes the following additions to the primary architecture. Every component not listed below is unchanged from Version 1.

#### Extended Phase Portrait

The Version 2 phase portrait extends <a href="#eq:portrait_v1" data-reference-type="eqref" data-reference="eq:portrait_v1">[eq:portrait_v1]</a> by adding limit cycle records: $$\mathcal{P}_i^{\mathrm{v2}}
= \mathcal{P}_i \cup
\left\{(\bar{x}^{(r)}, T^{(r)}, \rho_F^{(r)}, \mathbf{w}^{(r)}, \sigma^{(r)})
\right\}_{r \in \mathcal{I}_{\mathrm{LC}}},
$$ where $\bar{x}^{(r)} \in \mathbb{R}^{n_i}$ is the per-gene mean over one period, $T^{(r)}$ the period, $\rho_F^{(r)}$ the spectral radius of nontrivial Floquet multipliers (replacing “leading Floquet multiplier”), and $\mathbf{w}^{(r)} \in \mathbb{R}^{(2K+1) \times n_i}$ the Fourier coefficients of the raw gene waveforms under a canonical phase convention.

#### Extended Thomas Condition

Version 2 adds the second Thomas necessary condition: $$\text{No negative feedback loop in } G
\;\Rightarrow\; \text{no stable limit cycle.}
$$ This gates BVP continuation: SCCs without negative feedback loops never trigger BVP, concentrating the expensive computation on SCCs with the right topological prerequisites.

#### BVP Continuation for Limit Cycles

When a fixed point with near-Hopf signature is detected (ratio $|\alpha_{\mathrm{lead}}|/|\beta_{\mathrm{lead}}| < \tau$, where $\alpha_{\mathrm{lead}}$ is defined as in <a href="#eq:alpha_lead" data-reference-type="eqref" data-reference="eq:alpha_lead">[eq:alpha_lead]</a>: the complex pair with the smallest $|\alpha|$ among all complex pairs at that fixed point), BVP continuation is triggered: $$x(0) = x(T), \qquad
\int_0^T \langle \dot{x}_0(t), x(t) - x_0(t) \rangle\,dt = 0,
$$ with the Seydel reference orbit phase condition. BVP convergence confirms a limit cycle; BVP failure is inconclusive (not a proof of absence).

**Canonical phase convention for autonomous limit cycles:** After each BVP solve, shift the phase so that the sine coefficient of the fundamental harmonic of gene 1 is zero and the cosine coefficient is nonnegative. This is unique when gene 1’s fundamental amplitude is nonzero (generically true).

#### Extended Signal Construction

For a stable fixed point source branch, the signal remains a scalar (Version 1 unchanged). For a stable limit cycle source branch, the signal carries the full Fourier representation of the Hill-transformed waveform: $$s_r = (\hat{\mathbf{c}}_{ab}^{(r)}, \omega^{(r)}, \rho_F^{(r)}) \in \mathbb{R}^{2K+3},$$ where $\hat{\mathbf{c}}_{ab}^{(r)}$ are the Fourier coefficients of $H_{ab}(x_{i,a}^{(r)}(t))$ obtained by applying Hill pointwise to the predicted raw waveform $\hat{\mathbf{w}}_{i,a}^{(r)}$ and taking the DFT.

The Hill function is nonlinear, so this Fourier representation captures harmonic content that a mean-plus-amplitude description would miss.

#### Three Forcing Regimes and Nonautonomous Dynamics

When the source SCC is on a limit cycle, the receiving SCC is driven by a periodic signal and becomes nonautonomous. The three regimes are:

**Regime 1 (Version 1):** constant forcing. Receiving SCC autonomous.

**Regime 2:** periodic forcing at frequency $\omega_r$. Receiving SCC nonautonomous. Treated via state augmentation: introduce $\dot{\theta} = \omega_r$ to obtain an autonomous system on $\mathbb{R}^{n_j} \times S^1$. Attractors are forced periodic orbits and subharmonics — never fixed points (even under weak forcing).

**Regime 3** (incommensurate multi-frequency): out of scope. When multiple periodic upstream SCCs have incommensurate frequencies, the combined forcing is quasiperiodic. A runtime check flags these cases.

For multi-port Regime 2 with commensurate frequencies, relative phases $\{\phi_k\}$ between sources are explicit inputs (see the detailed treatment in the Version 2 companion document).

#### Regime-Split Physics Losses

For Regime 2 conditioned responses, the autonomous physics losses do not apply. Instead:

**Forced trajectory residual:** $\mathcal{L}_{\mathrm{physics}}^{\mathrm{forced}} =
\frac{1}{S}\sum_s \|\dot{\hat{x}}^{(r)}(t_s) - f(\hat{x}^{(r)}(t_s), u(t_s))\|^2$, evaluating the ODE residual along the predicted forced periodic orbit.

**Forced mean-drift:** $K_i^{\deg} \bar{x}_i = P_i \langle R_i(x(t)) \cdot u_i(t) \rangle_T$, which includes the external forcing term $u_i(t)$.

The Thomas exclusion loss does not transfer to Regime 2 (Thomas conditions apply to the autonomous unforced SCC, not to the nonautonomous forced system).

#### Extended Decoder

The decoder $D_\phi$ is extended to predict both FP and LC records. It takes an explicit regime indicator as conditioning input. For Regime 2, the output vocabulary consists of forced periodic orbit records (FPO) and subharmonic records (SHO), never FP records.

The decoder is trained on Corpus A (autonomous FP + LC), Corpus B Regime 1 (constant forced FP), and Corpus B Regime 2 (periodic forced FPO/SHO).

### Version 2 Corpus Extensions

###### Corpus A extension.

Add LC records to the ground truth: per-gene mean, period, Floquet spectral radius, and raw waveform Fourier coefficients under the canonical phase convention. Generated by BVP continuation triggered by near-Hopf signatures (using the $\alpha_{\mathrm{lead}}$ definition from <a href="#eq:alpha_lead" data-reference-type="eqref" data-reference="eq:alpha_lead">[eq:alpha_lead]</a>). BVP failure is inconclusive and recorded as a completeness flag.

###### Corpus B extension.

Add Regime 2 samples: periodic synthetic waveforms in Fourier series form, with relative phases for multi-port cases. Ground truth generated by augmented-state BVP <a href="#eq:bvp_auto_v2" data-reference-type="eqref" data-reference="eq:bvp_auto_v2">[eq:bvp_auto_v2]</a> with forcing phase as time origin. Stratify for entrainment, subharmonics, and bifurcation events.

###### Corpus C extension.

Add mixed-regime lineages: some SCCs in Regime 1 (constant forcing), some in Regime 2 (periodic forcing). Track typed branch lineages (FP vs. LC at each node). Relative phases tracked through the DAG.

### Summary: Version 1 vs. Version 2


| **Component**           | **Version 1**                   | **Version 2 addition**                       |
|:------------------------|:--------------------------------|:---------------------------------------------|
| Phase portrait objects  | Fixed points only               | \+ Limit cycles (LC)                         |
| Thomas conditions used  | Positive feedback only          | \+ Negative feedback                         |
| Corpus A numerics       | Newton–Raphson only             | \+ BVP continuation                          |
| Source branch types     | Stable FP only                  | \+ Stable LC                                 |
| Inter-SCC signal        | Scalar $s_r$                    | \+ Fourier series for LC                     |
| Receiving SCC dynamics  | Always autonomous               | \+ Nonautonomous (Regime 2)                  |
| Decoder output          | FP records only                 | \+ LC records, FPO, SHO                      |
| Physics losses          | Autonomous only                 | \+ Forced trajectory residual                |
| Phase convention        | N/A                             | Canonical autonomous + forcing-phase         |
| Relative phase          | N/A                             | Explicit for multi-port Regime 2             |
| Corpus cost             | Very cheap (no ODE integration) | \+ BVP solves (moderate)                     |
| Validation prerequisite | None                            | Version 1 validated                          |
| Empirical target        | Majority of real GRNs           | Circadian, cell cycle, metabolic oscillators |


### Mathematical Guarantees and Assumptions


| **Claim**                                                            | **Status**         | **Depends on**                                                                                                                      |
|:---------------------------------------------------------------------|:-------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| SCC condensation is a DAG                                            | Exact              | Graph theory                                                                                                                        |
| $\Omega$ positively invariant (strict)                               | Exact              | $(1+\varepsilon)$ bound <a href="#eq:ximax" data-reference-type="eqref" data-reference="eq:ximax">[eq:ximax]</a>                    |
| $f \neq 0$ on $\partial\Omega$                                       | Exact              | Strict invariance                                                                                                                   |
| $\deg(g, \Omega, 0) = 1$                                             | Exact              | Strict outward-pointing of $g=-f$                                                                                                   |
| Index sum $= 1$                                                      | Exact              | Hyperbolicity of fixed points                                                                                                       |
| Thomas (necessary only, one-directional)                             | Exact              | Established result                                                                                                                  |
| Bézout confirms finite FP count                                      | Conditional        | Integer Hill coefficients                                                                                                           |
| $K_{\mathrm{FP}}$ set from Corpus A statistics                       | Heuristic          | Pilot corpus histogram                                                                                                              |
| Hopf ratio $|\alpha_{\mathrm{lead}}|/|\beta_{\mathrm{lead}}| < \tau$ | **Heuristic only** | $\alpha_{\mathrm{lead}}$ = complex pair with smallest $|\alpha|$; not a mathematical Hopf condition                                 |
| BVP convergence confirms LC/FPO                                      | Numerical evidence | Newton solver convergence                                                                                                           |
| BVP failure rules out LC                                             | **False**          | Failure is inconclusive                                                                                                             |
| Monotone system signals are almost surely scalar                     | Exact (a.e.)       | Hirsch, strongly monotone                                                                                                           |
| Regime 2 has no FP attractors                                        | Exact              | Definition of nonautonomous system                                                                                                  |
| Incommensurate forcing in scope                                      | **Out of scope**   | Version 2 Assumption only                                                                                                           |
| Reconvergence: ancestry constrains forcing                           | Exact              | Deterministic ODE                                                                                                                   |
| Reconvergence: downstream attractor is free                          | Exact              | Initial conditions of intermediate SCCs                                                                                             |
| Edge density controls topology realism                               | **Design choice**  | Hyperparameter $c$ in <a href="#eq:edge_density" data-reference-type="eqref" data-reference="eq:edge_density">[eq:edge_density]</a> |

