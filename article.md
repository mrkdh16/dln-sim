<h1 class="article-title"><strong>Learning Dynamics of Deep Linear Networks</strong></h1>

<p class="article-subtitle">

  

Following <a href="https://arxiv.org/pdf/1312.6120" target="_blank">Saxe et al. (2014)</a> and <a href="https://arxiv.org/pdf/1810.10531" target="_blank">Saxe et al. (2018)</a>

  

</p>

## Introduction
**Deep linear networks** are great toy models. On the surface they seem trivial: stacking linear layers adds no representational power, since $W_L W_{L-1} \cdots W_1 = W_\text{total}$. But their *learning dynamics* are nonlinear and exhibit phenomena we see in real deep nonlinear networks: long plateaus in the loss, sudden rapid transitions, a preference for low-rank solutions, and an ordered curriculum over the structure in the data.

By studying deep linear networks, we can derive *exact solutions* to the learning dynamics and understand, rigorously, why these phenomena occur. The results turn out to approximate what we see in nonlinear networks remarkably well.
## Setup
#### The Toy Model and Task
Consider a 2-layer linear network $\hat{y} = W_2 W_1 x$ trained on examples $\{x^\mu, y^\mu\}$ ($\mu = 1, \ldots, P$) with mean-squared error loss:
$$
\mathcal{E} = \sum_{\mu=1}^{P} \frac{1}{2}\|y^\mu - W_2 W_1 x^\mu\|_2^2.
$$
Let the input $x^\mu \in \mathbb{R}^{N_1}$, output $y^\mu \in \mathbb{R}^{N_3}$, $W_1 \in \mathbb{R}^{N_2 \times N_1}$, $W_2 \in \mathbb{R}^{N_3 \times N_2}$, where $N_2$ is the hidden dimension.

Two matrices summarize all the statistical structure of the dataset used in learning. The **input-output correlation matrix** $\Sigma_{xy} \equiv \sum_\mu y^\mu {x^\mu}^\top \in \mathbb{R}^{N_3 \times N_1}$ encodes how each input dimension correlates with each output dimension across the whole dataset. The **input-input correlation matrix** $\Sigma_{xx} \equiv \sum_\mu x^\mu {x^\mu}^\top \in \mathbb{R}^{N_1 \times N_1}$ describes the spread and correlation structure of the inputs.
#### Gradient Flow
To understand the full trajectory of training, we pass from discrete gradient descent to continuous-time *gradient flow* (the limit of infinitesimally small learning rate):
$$
\begin{align*}
\frac{dW_1}{dt} &= W_2^\top\!\left(\Sigma_{xy} - W_2 W_1 \Sigma_{xx}\right) \tag{1}\\
\frac{dW_2}{dt} &= \left(\Sigma_{xy} - W_2 W_1 \Sigma_{xx}\right)W_1^\top. \tag{2}
\end{align*}
$$
The factor $\Sigma_{xy} - W_2 W_1 \Sigma_{xx}$ drives learning: it is zero at a fixed point, and its magnitude and structure determine what the network learns and in what order. Intuitively, $\Sigma_{xy}$ is the *target* the network is chasing, while $\Sigma_{xx}$ scales the gradient but doesn't directly generate the learning signal.
#### Whitened Inputs
Assume $\Sigma_{xx} = I$ (whitened inputs). Without this assumption, [the problem becomes NP-Hard](https://arxiv.org/abs/1012.0197). However, with this assumption, Equations (1) and (2) simplify to:
$$
\begin{align*}
\frac{dW_1}{dt} &= W_2^\top\!\left(\Sigma_{xy} - W_2 W_1\right) \tag{3}\\
\frac{dW_2}{dt} &= \left(\Sigma_{xy} - W_2 W_1\right)W_1^\top. \tag{4}
\end{align*}
$$
The learning task has been reduced to **matrix factorization**: gradient descent must find $W_2, W_1$ such that $W_2 W_1 \approx \Sigma_{xy}$. The simulator below lets you toggle this assumption to compare the two regimes.
## Solving for Exact Learning Dynamics
#### Rotating into the Natural Coordinate System
Equations (3) and (4) form a coupled nonlinear system—a change in any single weight affects every other weight. To solve them, we need to find a coordinate system in which the dynamics *decouple* into independent scalar equations.

Consider the **Singular Value Decomposition** (SVD) of $\Sigma_{xy}$: $\Sigma_{xy} = U S_* V^\top = \sum_{\alpha=1}^{r} s_\alpha\, \mathbf{u}_\alpha \mathbf{v}_\alpha^\top.$ Since the SVD modes of $\Sigma_{xy}$ drive learning, we rotate into their basis.

Define *rotated weight matrices*:
$$
\tilde{W}_1 \equiv W_1 V, \qquad \tilde{W}_2 \equiv U^\top W_2.
$$
Let $\mathbf{a}_\alpha$ denote column $\alpha$ of $\tilde{W}_1$ and $\mathbf{b}_\alpha$ denote row $\alpha$ of $\tilde{W}_2$. These are the *connectivity modes*: $\mathbf{a}_\alpha$ maps input mode $\alpha$ to hidden neurons, and $\mathbf{b}_\alpha$ maps hidden neurons to output mode $\alpha$.
#### Vector equations and the two subtasks of learning
Substituting into (3) and (4) and extracting individual columns/rows, the matrix equations become vector equations:
$$
\begin{align*}
\frac{d\mathbf{a}_\alpha}{dt} &= \mathbf{b}_\alpha(s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha) - \sum_{\gamma \neq \alpha} \mathbf{b}_\gamma(\mathbf{b}_\gamma \cdot \mathbf{a}_\alpha) \tag{5}\\
\frac{d\mathbf{b}_\alpha}{dt} &= (s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha)\,\mathbf{a}_\alpha - \sum_{\gamma \neq \alpha}(\mathbf{b}_\alpha \cdot \mathbf{a}_\gamma)\,\mathbf{a}_\gamma. \tag{6}
\end{align*}
$$
The cross-mode sums $\sum_{\gamma \neq \alpha}$ are the only source of coupling. They vanish exactly when distinct connectivity modes are *orthogonal* ($\mathbf{b}_\gamma \cdot \mathbf{a}_\alpha = 0$ for $\gamma \neq \alpha$). This reveals that gradient descent has two subtasks:
1. **Alignment** — rotate the connectivity modes so that distinct modes become orthogonal and cross-mode interference disappears
2. **Growth** — scale up the effective mode strengths $\mathbf{a}_\alpha \cdot \mathbf{b}_\alpha$ toward the target singular values $s_\alpha$. Empirically and [theoretically](https://arxiv.org/pdf/2111.00034), alignment happens very rapidly at the beginning of training, especially with small initialization (where random high-dimensional vectors are nearly orthogonal by default). We assume alignment has completed and focus entirely on the growth phase.
#### Decoupled Scalar Equations
Once aligned, each pair $(\mathbf{a}_\alpha, \mathbf{b}_\alpha)$ evolves along a shared direction $\mathbf{r}_\alpha$. Write:
$$
\mathbf{a}_\alpha(t) = a_\alpha(t)\,\mathbf{r}_\alpha, \qquad \mathbf{b}_\alpha(t) = b_\alpha(t)\,\mathbf{r}_\alpha,
$$
where $\{\mathbf{r}_\alpha\}$ are mutually orthogonal unit vectors. The cross-mode sums vanish, and Equations (5) and (6) collapse to independent *scalar* equations—one per mode:
$$
\begin{align*}
\frac{da_\alpha}{dt} &= b_\alpha(s_\alpha - a_\alpha b_\alpha) \tag{7}\\
\frac{db_\alpha}{dt} &= a_\alpha(s_\alpha - a_\alpha b_\alpha). \tag{8}
\end{align*}
$$
The effective singular value $\hat{s}_\alpha(t) \equiv a_\alpha(t) b_\alpha(t)$ represents the network's current strength along mode $\alpha$, i.e. the corresponding diagonal entry of $W_2 W_1$ in the SVD basis.
#### A separable ODE
Using the product rule on $\hat{s}_\alpha = a_\alpha b_\alpha$, and the approximation $a_\alpha \approx b_\alpha$ (valid for small, balanced initialization):
$$
\frac{d\hat{s}_\alpha}{dt} = b_\alpha^2(s_\alpha - \hat{s}_\alpha) + a_\alpha^2(s_\alpha - \hat{s}_\alpha) \approx 2\hat{s}_\alpha(s_\alpha - \hat{s}_\alpha). \tag{9}
$$
This is a standard separable ODE. Integrating with initial condition $\hat{s}_\alpha(0) = \hat{s}_\alpha^0$ gives the exact solution:
$$
\boxed{\hat{s}_\alpha(t) = \frac{s_\alpha}{1 + \displaystyle\left(\frac{s_\alpha}{\hat{s}_\alpha^0} - 1\right)e^{-2s_\alpha t}}.} \tag{10}
$$
This is precisely a **logistic (sigmoid) function** in time. The network's effective singular values follow S-shaped curves, transitioning from near-zero to near $s_\alpha$.

Since (via Eq. 7) the entire weight product evolves as $W_2 W_1(t) = U\,\hat{S}(t)\,V^\top$, Equation (10) gives a complete analytical description of every parameter in the network throughout training.

Note that in deeper linear networks, although we can obtain ODEs similar to Equation (9), we can't solve them analytically. In the simulation below, we numerically integrate to obtain solutions.
## Key Findings
#### 1. Ordered learning by singular value strength
The time for mode $\alpha$ to reach its target strength scales as $O(1/s_\alpha)$. Modes with larger singular values—corresponding to more prominent, statistically dominant structure in the data—are learned first. Subtle distinctions with small $s_\alpha$ take much longer to internalize. This phenomenon, where a network first grasps broad categorical structure before refining fine-grained detail, is observed empirically in nonlinear networks as well.
#### 2. Sigmoid dynamics are a signature of depth
In a shallow (1-layer) linear network, each mode's effective singular value approaches its target *exponentially*: $\hat{s}_\alpha(t) = s_\alpha + (\hat{s}_\alpha^0 - s_\alpha)e^{-2s_\alpha t}$. The logistic S-curve—slow start, rapid transition, plateau—is a direct mathematical consequence of depth. It requires no nonlinear activation functions; depth alone induces this behavior through the coupling of the weight matrices in the product $W_2 W_1$.
#### 3. Plateaus and sudden transitions are inevitable
When $s_\alpha \gg \hat{s}_\alpha^0$ (small initialization relative to the signal strength), the sigmoid transition is very sharp. The network spends a long time in a near-flat plateau before rapidly learning the mode. This explains why training loss curves often show long periods of little apparent progress followed by sudden drops—not a failure of the optimizer, but an intrinsic mathematical property of gradient descent in deep architectures.
<div class="findings-note">

<p>

The figure below (from Saxe et al. 2014) compares these theoretical predictions (red) against empirical measurements in deep linear networks (green) and nonlinear networks (blue). The agreement is striking.

</p>

<div class="article-figure">

<img src="Screenshot 2026-02-11 at 12.29.19 AM.png" alt="Theoretical vs empirical singular value trajectories" width=350>

<figcaption>

From Saxe et al. 2014: "Dynamics of learning in a [two] layer neural network. Curves show the strength of the network’s representation of seven modes of the input-output correlation matrix over the course of learning. Red traces show analytical curves from Eqn. [10]. Blue traces show simulation of full dynamics of a [two layer] linear network ... from small random initial conditions. Green traces show simulation of a nonlinear [two] layer network with tanh activation functions."

</figcaption>

</div>

</div>

<div class="scroll-prompt">

<div class="scroll-prompt-line"></div>

<p>Try the interactive simulator below — configure the network, start training, and watch singular values evolve</p>

<div class="scroll-arrow">↓</div>

</div>