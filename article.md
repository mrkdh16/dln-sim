## Setup
#### The Toy Model and Task
Consider a 2-layer linear network $\hat{y} = W_2 W_1 x$ trained on examples ${x^\mu, y^\mu}$ ($\mu = 1, \ldots, P$) with mean-squared error loss: $$

\mathcal{E} = \sum_{\mu=1}^{P} \frac{1}{2}|y^\mu - W_2 W_1 x^\mu|_2^2.

$$ Let the input $x^\mu \in \mathbb{R}^{N_1}$, output $y^\mu \in \mathbb{R}^{N_3}$, $W_1 \in \mathbb{R}^{N_2 \times N_1}$, $W_2 \in \mathbb{R}^{N_3 \times N_2}$, where $N_2$ is the hidden dimension.

Two matrices summarize all the statistical structure of the dataset used in learning. The **input-output correlation matrix** $\Sigma_{xy} \equiv \sum_\mu y^\mu {x^\mu}^\top \in \mathbb{R}^{N_3 \times N_1}$ encodes how each input dimension correlates with each output dimension across the whole dataset. The **input-input correlation matrix** $\Sigma_{xx} \equiv \sum_\mu x^\mu {x^\mu}^\top \in \mathbb{R}^{N_1 \times N_1}$ describes the spread and correlation structure of the inputs.
#### Gradient Flow
To understand the full trajectory of training, we pass from discrete gradient descent to continuous-time _gradient flow_ (the limit of infinitesimally small learning rate): $$

\begin{align*}

\frac{dW_1}{dt} &= - \frac{\partial \mathcal{E}}{\partial W_1} = W_2^\top\left(\Sigma_{xy} - W_2 W_1 \Sigma_{xx}\right) \tag{1}\\

\frac{dW_2}{dt} &= -\frac{\partial \mathcal{E}}{\partial W_2} = \left(\Sigma_{xy} - W_2 W_1 \Sigma_{xx}\right)W_1^\top. \tag{2}

\end{align*}

$$The factor $\Sigma_{xy} - W_2 W_1 \Sigma_{xx}$ drives learning: it is zero at a fixed point, and its magnitude and structure determine what the network learns and in what order. Intuitively, $\Sigma_{xy}$ is the _target_ the network is chasing, while $\Sigma_{xx}$ scales the gradient but doesn't directly generate the learning signal.
#### Whitened Inputs
Assume $\Sigma_{xx} = I$ (whitened inputs). With this assumption, Equations (1) and (2) simplify to: 
$$
\begin{align*}

\frac{dW_1}{dt} &= W_2^\top\left(\Sigma_{xy} - W_2 W_1\right) \tag{3}\\

\frac{dW_2}{dt} &= \left(\Sigma_{xy} - W_2 W_1\right)W_1^\top. \tag{4}

\end{align*}
$$The learning task has been reduced to **matrix factorization**: gradient descent must find $W_2, W_1$ such that $W_2 W_1 \approx \Sigma_{xy}$. The simulator below lets you toggle this assumption to compare the two regimes.
## Solving for Exact Learning Dynamics
#### Rotating into the Natural Coordinate System
Equations (3) and (4) form a coupled nonlinear system—a change in any single weight affects every other weight. To solve them, we need to find a coordinate system in which the dynamics _decouple_ into independent scalar equations.

Consider the **Singular Value Decomposition** (SVD) of the input-output correlation matrix $\Sigma_{xy} = U S_* V^\top = \sum_{\alpha=1}^{r} s_\alpha, \mathbf{u}_\alpha \mathbf{v}_\alpha^\top.$ Since the SVD modes of $\Sigma_{xy}$ drive learning, we rotate into their basis.

Define _rotated weight matrices_: $$

\tilde{W}_1 \equiv W_1 V, \qquad \tilde{W}_2 \equiv U^\top W_2.

$$ Let $\mathbf{a}_\alpha$ denote column $\alpha$ of $\tilde{W}_1$ and $\mathbf{b}_\alpha$ denote row $\alpha$ of $\tilde{W}_2$. These are the _connectivity modes_: $\mathbf{a}_\alpha$ maps input mode $\alpha$ to hidden neurons, and $\mathbf{b}_\alpha$ maps hidden neurons to output mode $\alpha$.
#### Vector equations and the two subtasks of learning
Substituting into (3) and (4) and extracting individual columns/rows, the matrix equations become vector equations: 
$$
\begin{align*}

\frac{d\mathbf{a}_\alpha}{dt} &= \mathbf{b}_\alpha(s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha) - \sum_{\gamma \neq \alpha} \mathbf{b}_\gamma(\mathbf{b}_\gamma \cdot \mathbf{a}_\alpha) \tag{5}\\

\frac{d\mathbf{b}_\alpha}{dt} &= (s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha),\mathbf{a}_\alpha - \sum_{\gamma \neq \alpha}(\mathbf{b}_\alpha \cdot \mathbf{a}_\gamma),\mathbf{a}_\gamma. \tag{6}

\end{align*}

$$Equations (5) and (6) reveal the structure of the learning problem. At a fixed point, both derivatives must vanish. Consider Equation (5): it has two terms. The **first term**, $\mathbf{b}_\alpha(s_\alpha - \mathbf{b}_\alpha \cdot \mathbf{a}_\alpha)$, vanishes when $\mathbf{b}_\alpha \cdot \mathbf{a}_\alpha = s_\alpha$; i.e. when the effective strength of mode $\alpha$ has grown to match its target singular value. The **second term**, $\sum_{\gamma \neq \alpha} \mathbf{b}_\gamma(\mathbf{b}_\gamma \cdot \mathbf{a}_\alpha)$, vanishes when distinct connectivity modes are orthogonal: $\mathbf{b}_\gamma \cdot \mathbf{a}_\alpha = 0$ for all $\gamma \neq \alpha$. (The same logic applies to Equation (6) with the roles of $\mathbf{a}$ and $\mathbf{b}$ exchanged.) In other words, the network wants to achieve $\mathbf{b}_ \beta \cdot \mathbf{a}_\alpha \approx s_\alpha \delta_{\alpha\beta}$, and gradient descent must solve two subtasks to get there:

1. **Alignment** — rotate the connectivity modes so that distinct modes become orthogonal and cross-mode interference disappears (driving the second term to zero)
2. **Growth** — scale up the effective mode strengths $\mathbf{a}_{\alpha}\cdot \mathbf{b}_{\alpha}$ toward the target singular values $s_{\alpha}$ (driving the first term to zero)

Empirically and [theoretically](https://arxiv.org/pdf/2111.00034), alignment happens very rapidly at the beginning of training, especially with small initialization (where random high-dimensional vectors are nearly orthogonal by default). Once alignment is achieved, the coupled vector equations collapse into independent scalar equations—one per mode—and we can solve for the full time course of learning. We assume alignment has completed and focus entirely on the growth phase. The simulator below lets you toggle this assumption.
#### Decoupled Scalar Equations
Once aligned, each pair $(\mathbf{a}_\alpha, \mathbf{b}_\alpha)$ evolves along a shared direction $\mathbf{r}_\alpha$. Write:
$$

\mathbf{a}_\alpha(t) = p_\alpha(t),\mathbf{r}_\alpha, \qquad \mathbf{b}_\alpha(t) = q_\alpha(t),\mathbf{r}_\alpha,

$$
where $p_\alpha$ and $q_\alpha$ are the scalar projections of $\mathbf{a}_\alpha$ and $\mathbf{b}_\alpha$ onto their shared direction, and ${\mathbf{r}_\alpha}$ are mutually orthogonal unit vectors. The cross-mode sums vanish, and Equations (5) and (6) collapse to independent _scalar_ equations—one per mode:
$$

\begin{align*}

\frac{dp_\alpha}{dt} &= q_\alpha(s_\alpha - p_\alpha q_\alpha) \tag{7}\\

\frac{dq_\alpha}{dt} &= p_\alpha(s_\alpha - p_\alpha q_\alpha). \tag{8}

\end{align*}

$$
The effective singular value $\hat{s}_\alpha(t) \equiv p_\alpha(t) q_\alpha(t)$ represents the network's current strength along mode $\alpha$, i.e. the corresponding diagonal entry of $W_2 W_1$ in the SVD basis.
#### A Separable ODE
Using the product rule on $\hat{s}_\alpha = p_\alpha q_\alpha$, and the approximation $p_\alpha \approx q_\alpha$ (valid for small, balanced initialization):
$$

\frac{d\hat{s}_\alpha}{dt} = q_\alpha^2(s_\alpha - \hat{s}_\alpha) + p_\alpha^2(s_\alpha - \hat{s}_\alpha) \approx 2\hat{s}_\alpha(s_\alpha - \hat{s}_\alpha). \tag{9}

$$
This is a standard separable ODE. Integrating with initial condition $\hat{s}_\alpha(0) = \hat{s}_\alpha^0$ gives the exact solution:
$$

\boxed{\hat{s}_\alpha(t) = \frac{s_\alpha}{1 + \displaystyle\left(\frac{s_\alpha}{\hat{s}_\alpha^0} - 1\right)e^{-2s_\alpha t}}.} \tag{10}

$$
This is precisely a **logistic (sigmoid) function** in time. The network's effective singular values follow S-shaped curves, transitioning from near-zero to near $s_\alpha$.

Since (via Eq. 7) the entire weight product evolves as $W_2 W_1(t) = U,\hat{S}(t),V^\top$, Equation (10) gives a complete analytical description of every parameter in the network throughout training.
## Deeper Networks
The analysis above was carried out for a 2-layer network, but the same framework extends naturally to networks of arbitrary depth $L$. In a network with $L$ weight matrices $W_1, \ldots, W_{L}$, the same decoupling into independent connectivity modes applies under the analogous initial conditions. Each mode is then described not by two scalars $(p_\alpha, q_\alpha)$ but by $L$ scalars $a^1_\alpha, \ldots, a^{L}_\alpha$, one per weight matrix, and the effective singular value becomes the product $\hat{s}_\alpha = \prod_{i=1}^{L} a^i_\alpha$.

On the symmetric submanifold where all per-layer amplitudes are equal ($a^i_\alpha = a_\alpha$ for all $i$), the ODE for the overall mode strength generalizes from Equation (9) to
$$

\frac{d\hat{s}_\alpha}{dt} = L\,\hat{s}_\alpha^{\,2 - 2/L}(s_\alpha - \hat{s}_\alpha). \tag{11}

$$
The structure is recognizably similar: the same $(s_\alpha - \hat{s}_\alpha)$ driving term appears, and learning is still ordered by singular value strength. What changes is the power of $\hat{s}_\alpha$ multiplying this term. For $L=2$, the exponent $2 - 2/L$ equals 1 and we recover $2\hat{s}_\alpha(s_\alpha - \hat{s}_\alpha)$, which is the separable logistic equation we solved in closed form above. But for any deeper network, the exponent is nonzero and the ODE becomes much more complicated. To obtain the time course of learning in deeper networks, it's easiest to numerically integrate Equation (11). The simulator below does exactly this, letting you vary network depth and observe how the sigmoid transitions sharpen and the learning dynamics change.
## Key Findings
#### 1. Ordered learning by singular value strength
The time for mode $\alpha$ to reach its target strength scales as $O(1/s_\alpha)$. Modes with larger singular values—corresponding to more prominent, statistically dominant structure in the data—are learned first. Subtle distinctions with small $s_\alpha$ take much longer to internalize. This phenomenon, where a network first grasps broad categorical structure before refining fine-grained detail, is observed empirically in nonlinear networks as well.
#### 2. Sigmoidal dynamics are a signature of depth
In a shallow (1-layer) linear network, each mode's effective singular value approaches its target _exponentially_: $\hat{s}_\alpha(t) = s_\alpha + (\hat{s}_\alpha^0 - s_\alpha)e^{-2s_\alpha t}$. The logistic S-curve—slow start, rapid transition, plateau—is a direct mathematical consequence of depth. It requires no nonlinear activation functions; depth alone induces this behavior through the coupling of the weight matrices in the product $W_2 W_1$.
#### 3. Plateaus and sudden transitions are inevitable
When $s_\alpha \gg \hat{s}_\alpha^0$ (small initialization relative to the signal strength), the sigmoid transition is very sharp. The network spends a long time in a near-flat plateau before rapidly learning the mode. This explains why training loss curves often show long periods of little apparent progress followed by sudden drops. It's an intrinsic mathematical property of gradient descent in deep architectures.

<div class="scroll-prompt">

<div class="scroll-prompt-line"></div>

<p>Try the interactive simulator below — configure the network, start training, and watch singular values evolve</p>

<div class="scroll-arrow">↓</div>

</div>