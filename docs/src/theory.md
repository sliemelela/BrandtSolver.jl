# Theory
To understand how the package works, we treat the algorithm proposed by [Brandt_etal_2005](@cite) as our foundation.
While the original authors suggest the use of Taylor expansions to approximate the value function,
most implementations are limited to a second-order (mean-variance) approximation.
This package extends that logic by providing the generalized expressions and implementation for a Taylor expansion of any order $k$.
By deriving the multinomial expansion of the budget constraint and the associated high-order derivatives of the value function, this tool allows for greater precision in capturing non-normalities and higher-order moments in asset returns.


## Goal of the algorithm
In this package we are treating a terminal wealth optimization problem.
More specifically, in this package we consider portfolio choice problems at timesteps
$n = 1, 2, \ldots, M$, where $M + 1$ is some terminal timestep.
This portfolio choice problem at timestep $n$ is defined by an investor who maximizes the expected utility
of their wealth at the terminal timestep $M + 1$ by trading $N$ risky assets and a risk-free asset (cash).
Formally the investor's problem at timestep $n$ is
```math
    V_n(W_n, Z_n)
    = \max_{\{\omega_s\}_{m = n}^{M}} \mathbb{E}_n[u(W_{M + 1})]
```
subject to the sequence of budget constraints
```math
    W_{m + 1} = W_m (\omega_m^\top R^e_{m + 1} + R_{m + 1})
```
for all $m \geq n$.
Here $R^e_{m + 1}$ can be interpreted as the excess return of the risky assets over the risk-ree
asset, and $R_{m + 1}$ is the gross return of other processes that _may_ depend on wealth
$W_m$.
Furthermore, $\{\omega_s\}_{m=n}^{M}$ is the sequence of portfolio weights chosen at times
$m = n, \ldots, M$ and $u$ is the investor's utility function.
The process $Z_n$ is a vector of state variables that are relevant for the investor's decision making.
Lastly, the function $u$ denotes the utility function of the investor.
The goal of this package is to find $\{\omega_m\}_{m=1}^{M}$.

### Extension: Wealth-Dependent Returns
A key feature of this implementation is that the gross return $R_{m+1}$ is not restricted to be exogenous.
We allow $R_{m+1}$ to depend on the current level of wealth $W_m$ through the following structure:
```math
    R_{m+1} = X_m + \frac{Y_m}{W_m}
```
where $X_m$ and $Y_m$ are functions of state variables contained in $Z_m$.
This formulation is particularly powerful as it allows the algorithm to incorporate
**non-tradeable income or fixed costs**.
For instance, if $Y_m$ represents labor income, the budget constraint correctly captures that
income is added to wealth regardless of the portfolio choice
$\omega_m$.

#### Example
Consider an investor who receives a stochastic labor income $O_n$ at each timestep and saves a
proportion $p \in [0,1]$ of that income.
Let $R^f_n$ be the gross risk-free rate.
The wealth at time $n+1$ is:
```math
    W_{n+1} = W_n (\omega_n^\top R^e_{n+1} + R^f_n) + p O_n
```
By setting $X_n = R^f_n$ and $Y_n = p O_n$,
this matches our budget constraint $W_{n+1} = W_n (\omega_n^\top R^e_{n+1} + R_{n+1})$.



## Solution of the algorithm
The goal of this section is to give the final solution needed to set up the algorithm.
We acknowledge that the result can look quite daunting and so it is not the expectation that the
reader can immediately make sense of why the solution looks like it does.
Despite this, it can help to first see where we are working towards before explaining how it is
derived. Let us first start by specifying the notation.


### Notation
Let $\omega_m = (\omega^1_m, \ldots, \omega^N_m)$ be the components of the portfolio weights
at some timestep $m$ and write $R^e_{m + 1} = (R^{e, 1}_{m + 1}, \ldots, R^{e, N}_{m + 1})$
for the components for the excess return vector corresponding to each asset that is traded.
If $\omega_m$ is chosen optimally, we decorate it with a $\star$ in the superscript, i.e.
$\omega_m^{\star}$ is the vector of optimal portfolio weights at timestep $m$.
Furthermore, let us assume that the current timestep is denoted by $n$ and let us also define
```math
    \hat W_{M + 1} = W_n R_{n + 1}
        \prod_{m = n + 1}^{M} ((\omega_m^{\star})^\top R^e_{m + 1} + R_{m + 1}).
```

### Result
We assume that the current timestep is denoted by $n$ and that all optimal portfolio weights at
times $m > n$ are known, i.e. $\{\omega_m^\star\}_{m = n + 1}^{M}$ are known.
Assuming the value function $V$ is infinitely differentiable in the first argument, the optimal
portfolio weights $\omega_n^\star$ is the limit of the solutions of the equation indexed by
$k$
```math
\begin{aligned}
    \sum_{r = 1}^{k} \frac{W_t^{r - 1}}{(r - 1)!}  &\Biggl(\sum_{k_1 + \ldots + k_N = r - 1}
    \binom{r - 1}{k_1, \ldots, k_N} \prod_{i=1}^N (\omega_n^i)^{k_i} \times \\
   &\mathbb{E}_n \left[\partial^{r} u(\hat W_{M + 1})
      \prod_{m = n + 1}^{M} ((\omega_m^\star)^\top R^e_{m + 1} + X_{m})^r
      \prod_{i=1}^N (R^i_{n + 1})^{k_i} R^e_{n+1}\right]\Biggr) = 0,
\end{aligned}
```
where $\binom{p}{k_1,\ldots, k_N}$ is the multinomial given by
```math
  \binom{p}{k_1,\ldots, k_N} = \frac{p!}{k_1 ! k_2 ! \cdots k_N !}.
```

Given this result, the pragmatic reader may come up with some reasonable questions:
- How should one choose $k$?
- How should one calculate the conditional expectation value $\mathbb{E}_n[\cdot]$?
- How should one find the optimal portfolio weights $\omega_m^\star$ for $m > n$?

The short and concise answers are:
- The higher $k$, the better. In our experience $4$ suffices for most applications.
- This is done by means of regressions. This can be unstable however.
    Dropping part of the data generated by the integrand $\cdot$ in $\mathbb{E}_n[\cdot]$ seems to partly
    solve this issue.
- This is done by first solving for $\omega_M^\star$. You can then recursively find the solution for
    any time.

If the above does not make sense, do not fret.
This will soon all make sense once we treat the derivation of the above.

## Deriving the solution: so, how do we get there?
The (rough) outline of the derivation is as follows:
1. We prove the value function satisfies the Bellman equation
```math
V_n(W_n, Z_n) = \max_{\omega_n} \mathbb{E}_n[V_{n + 1}(W_{n + 1}, Z_{n + 1})]
```
2. We derive the Taylor expansion the value function $V_{n + 1}$ in the conditional expectation.
    This gives a new expression for $V_n$.
    _Note_: This is where the $k$ came from in the [solution](#result).
3. Using the new expression for $V_n$ we derive the first order conditions of finding the optimal
    $\omega_n$.
4. We manipulate the first order conditions so that all unknown quantities are replaced by
    known quantities.

### The Bellman equation
The value function satisfies the Bellman equation
```math
  V_n(W_n, Z_n) = \max_{\omega_n} \mathbb{E}_n[V_{n + 1}(W_{n + 1}, Z_{n + 1})]
```
with terminal condition $V_{M + 1} (W_{M + 1}, Z_{M + 1}) = u(W_{M + 1})$.

#### Proof
We see that
```math
\begin{aligned}
    V_n(W_n, Z_n)
    &= \max_{\{\omega_m\}_{m = n}^{M}} \mathbb{E}_n\left[ u(W_{M + 1}) \right]\\
    &= \max_{\omega_n} \max_{\{\omega_m\}_{m = n + 1}^{M}} \mathbb{E}_n\left[ u(W_{M + 1}) \right] \\
    &= \max_{\omega_n} \max_{\{\omega_m\}_{m = n + 1}^{M}}
        \mathbb{E}_n\left[  \mathbb{E}_{n + 1}\left[ u(W_{M + 1}) \right] \right] \\
    &= \max_{\omega_n} \mathbb{E}_n\left[ \max_{\{\omega_m\}_{m = n + 1}^{M}}
        \mathbb{E}_{n + 1}\left[ u(W_{M + 1}) \right] \right] \\
    &= \max_{\omega_n} \mathbb{E}_n\left[ V_{n + 1}(W_{n + 1}, Z_{n + 1}) \right].
\end{aligned}
```
The first equality follows from the definition of the value function.
The second equality follows from separating the maximization over $\omega_t$ and the remaining
maximization over $\{\omega_m\}_{m = n + 1}^{M}$.
The third equality follows from the law of iterated expectations.
The fourth equality follows from the fact that the maximization over
$\{\omega_m\}_{m = n + 1}^{M}$ is independent of the choice of $\omega_n$ and can thus be moved inside the expectation.
And the last equality follows from the definition of the value function at time $t + 1$.

### Taylor expanding the value function
Let us assume that the value function $V$ is $C^{k + 1}$ in the first argument for $k \geq 2$.
Then the value function (using the Bellman equation) satisfies
```math
\begin{aligned}
    V_n(W_n, Z_n)
    &= \max_{\omega_n} \Biggl\{ \sum_{r = 0}^{k} \frac{W_n^r}{r!} \mathbb{E}_n \left[ \partial_1^r V_{n+1}(W_n R_{n+1}, Z_{n+1}) (\omega_n^\top R^e_{n+1})^r \right] \\
    &+ \frac{W_n^{k+1}}{(k+1)!} \mathbb{E}_n \left[ \partial_1^{k+1} V_{n+1}(\xi, Z_{n+1}) (\omega_n^\top R^e_{n+1})^{k+1} \right]
    \Biggr\}
\end{aligned}
```
for some $\xi$ between $W_n R_{n + 1}$ and $W_n (\omega_n^\top R^e_{n + 1} + R_{n + 1})$.
Here we assume that all moments exist and are finite.

#### Proof
Let us consider the mapping $f(x) = V_{n + 1} (x, Z_{n + 1})$ for fixed $n$ and $Z_{n + 1}$.
Using Taylor's theorem, we have that
```math
    f(x) = \sum_{r = 0}^k \frac{1}{r!} f^{(r)}(x_0)(x - x_0)^r + \frac{f^{(k + 1)}(\xi)}{(k+1)!}(x - x_0)^{k + 1}
```
for some $\xi$ between $x$ and $x_0$.
Taking $x_0 = W_n R_{n + 1}$ and $x = W_{n + 1}$, and noting that
$W_{n + 1} - W_n R_{n + 1} = W_n \omega_n^\top R^e_{n + 1}$ by the budget constraint,
the result follows.

### Finding the first order conditions
The next is to solve the static maximization problem in the Taylor expansion.
The next proposition provides the final equation for this.

Let $\omega_n = (\omega_n^1, \dots, \omega_n^N)$ be the components of the portfolio weights at time $n$.
Similarly, write $R^e_{n+1} = (R_{n+1}^{e,1}, \dots, R_{n+1}^{e,N})$.
The first order conditions (FOC) are given by:
```math
   \sum_{r = 1}^{k} \frac{W_n^{r - 1}}{(r - 1)!} \mathbb{E}_n \left[ \partial_1^r V_{n + 1}(W_n R_{n + 1}, Z_{n + 1}) (\omega_n^\top R^e_{n + 1})^{r - 1} R^e_{n + 1} \right] + \frac{W_n^k}{k!} \mathbb{E}_n \left[ \partial_1^{k + 1} V_{n + 1}(\xi, Z_{n + 1}) (\omega_n^\top R^e_{n + 1})^{k} R^e_{n + 1} \right] = 0
```
or alternatively:
```math
\begin{aligned}
   \sum_{r = 1}^{k} \frac{W_n^{r - 1}}{(r - 1)!}  &\Biggl(\sum_{k_1 + \dots + k_N = r - 1} \binom{r - 1}{k_1, \dots, k_N} \prod_{i=1}^N (\omega_n^i)^{k_i} \times \\
   &\mathbb{E}_n \left[ \partial_1^r V_{n + 1}(W_n R_{n + 1}, Z_{n + 1}) \prod_{i=1}^N (R^{e, i}_{n + 1})^{k_i} R^e_{n+1} \right]\Biggr) \\
    &+ \frac{W_n^k}{k!} \mathbb{E}_n \left[ \partial_1^{k + 1} V_{n + 1}(\xi, Z_{n + 1}) (\omega_n^\top R^e_{n + 1})^{k} R^e_{n + 1} \right] = 0.
\end{aligned}
```

#### Proof
To derive the FOC, we take the static maximization problem from the Taylor expansion,
take the derivative with respect to $\omega_n$, and set it to $0$.
After this, we divide both sides by $W_n$. This yields the first result.

We now note that for arbitrary $r \in \mathbb{N}$, it holds by the multinomial theorem that:
```math
(\omega_n^\top R^e_{n + 1})^{r-1} = \sum_{k_1 + \dots + k_N = r-1}
    \binom{r-1}{k_1, \dots, k_N} \prod_{i=1}^N (\omega_n^i R^{e, i}_{n + 1})^{k_i}
```
where $\binom{p}{k_1, \dots, k_N}$ is the multinomial coefficient.
Using this identity and the linearity of the expectation, we find the second result.

### Finding an alternative way of writing the value function
The last problem we ought to solve is the problem of having no expression for the derivatives of the
value function $\partial^r_1 V_{n + 1}(W_n R_{n+1}, Z_{n + 1})$. To that end, we note the following lemma.

Let $\{\omega^\star_m\}_{m = n + 1}^{M}$ be the optimal sequence of portfolio weights at times $m = n + 1, \dots, M$,
and denote
```math
\hat{W}_{M+1} = W_n R_{n + 1} \prod_{m = n+1}^{M} ((\omega_m^\star)^\top R^e_{m + 1} + R_{m + 1}).
```
Then
```math
\partial^{r}_1 V_{n + 1}(W_n R_{n + 1}, Z_{n + 1})
= \mathbb{E}_{n + 1} \left[ u^{(r)}(\hat{W}_{M+1}) \prod_{m = n + 1}^{M} ((\omega_m^\star)^\top R^e_{m + 1} + X_m)^r \right].
```


#### Proof
Since we are interested in taking the derivative of the first argument of the value function
$V_{n + 1}$, we need to make the dependence of $W_{n + 1}$ (the first argument) as clear as possible.
For that reason we will first rewrite the budget constraint after which we show the above.

Recall that the budget constraint is given by
```math
    W_{m + 1} = W_m (\omega_m^\top R^e_{m + 1} + R_{m + 1})
```
where $R_{m + 1} = X_{m} + Y_m / W_m$.
Defining $G_{m + 1} = (\omega_m^\top R^e_{m + 1} + X_m)$ and rewriting the budget constraint to
```math
    W_{m + 1} = W_m (\omega_m^\top R^e_{m + 1} + X_m) + Y_m,
```
we can write
```math
    W_{M + 1} = W_{n + 1} \left(\prod_{m = n + 1}^{M} G_{m + 1}\right)
    + \sum_{m = n + 1}^{M} Y_{m} \left(\prod_{p = m + 1}^{M} G_{p + 1}\right)
```
where $n < M$ is some arbitrary timestep.
The critical reader might demand a proof of this. This can be found in the appendix.
Using this, we now note that
```math
\frac{\partial W_{M + 1}}{\partial W_{n + 1}} =  \prod_{m = n + 1}^{M} G_{m + 1}.
```

Using the definition of the value function and assuming $\{\omega^\star_m\}$ is the optimal sequence, we have
```math
    V_{n + 1}(x, Z_{n + 1})
        = \mathbb{E}_{n + 1} \left[ u \left( W_{M + 1}\right) \right],
```
where
```math
 W_{M + 1} = x \left(\prod_{m = n + 1}^{M} G_{m + 1}\right)
    + \sum_{m = n + 1}^{M} Y_{m} \left(\prod_{p = m + 1}^{M} G_{p + 1}\right)
```
Using the Leibniz rule to differentiate under the expectation sign, we compute the $r$-th order derivative of the value function with respect to its first argument $x$ which is
```math
    \partial_1^{r} V_{n + 1}(x, Z_{n + 1}) = \mathbb{E}_{n + 1} \left[ \frac{\partial^r}{\partial x^r} u(W_{M + 1}) \right].
```
By the chain rule, and noting that $W_{M+1}$ is an affine function of $x$, the first derivative is:$$\frac{\partial}{\partial x} u(W_{M+1}) = u'(W_{M+1}) \frac{\partial W_{M+1}}{\partial x} = u'(W_{M+1}) \left( \prod_{m = n + 1}^{M} G_{m+1} \right)$$Since the term $\prod G_{m+1}$ does not depend on $x$, subsequent derivatives follow a power-rule pattern for the marginal growth term. For any $r \geq 1$, we have
```math
\frac{\partial^r}{\partial x^r} u(W_{M+1}) = u^{(r)}(W_{M+1}) \left( \prod_{m = n + 1}^{M} G_{m+1} \right)^r.
```
Substituting the expression for the terminal wealth evaluated at the specific point $x = W_n R_{n+1}$, which we denote as $\hat{W}_{M+1}$, gives
```math
\partial_1^{r} V_{n + 1}(W_n R_{n+1}, Z_{n + 1}) = \mathbb{E}_{n + 1} \left[ u^{(r)}(\hat{W}_{M+1}) \left( \prod_{m = n + 1}^{M} G_{m+1} \right)^r \right].
```
This yields the desired result, expressing the derivative of the value function solely in terms of the utility function's derivatives, the realized future wealth, and the marginal returns on investment.



### Final result
Using this, we can now reformulate the FOC.

The first order conditions (FOC) are given by:
```math
\begin{aligned}
   \sum_{r = 1}^{k} \frac{W_n^{r - 1}}{(r - 1)!}  &\Biggl(\sum_{k_1 + \dots + k_N = r - 1} \binom{r - 1}{k_1, \dots, k_N} \prod_{i=1}^N (\omega_n^i)^{k_i} \times \\
   &\mathbb{E}_n \left[ u^{(r)}(\hat{W}_{M+1}) \prod_{m = n + 1}^{M} ((\omega_m^\star)^\top R^e_{m + 1} + X_{m + 1})^r \prod_{i=1}^N (R^{e, i}_{n + 1})^{k_i} R^e_{n+1} \right]\Biggr) \\
    &+ \text{Remainder} = 0.
\end{aligned}
```


## High level details on implementation
1) The processes $R^e_{n + 1}, Z_{n}$ are generated (which includes $X_n$ and $Y_n$)
2) At time $n$, it is assumed all future portfolio choices $\omega_m^\star$ are known
3)
### Initial guess for polynomial equation
In most polynomial solvers, it is necessary to provide an initial guess.
To that end, we consider the $2$-nd order expansion ($k=2$), which yields the linear FOC:
```math
  a_n + B_n \omega_n = 0 \implies \omega_n = -B_n^{-1} a_n
```
where $a_n$ is a vector and $B_n$ is a matrix with columns $b_{i, n}$ given by
```math
\begin{aligned}
  a_{n} &= \mathbb{E}_n \left[ u'(\hat{W}_{M+1}) \prod_{m = n + 1}^{M} ((\omega_m^\star)^\top R^e_{m + 1} + X_m) R^e_{n+1} \right], \\
  b_{i, n} &= \mathbb{E}_n \left[ u''(\hat{W}_{M+1}) \prod_{m = n + 1}^{M} ((\omega_m^\star)^\top R^e_{m + 1} + X_m)^2 R^{e, i}_{n + 1} R^e_{n+1} \right].
\end{aligned}
```

###
The approximation $W_s \approx \hat{W}_s$ for $s > n$ effectively assumes that the
marginal impact of the current portfolio choice $\omega_n$ on the future wealth-dependent
returns $R_{s+1}$ is negligible. Given that $\omega_n^\top R^e_{n+1}$ represents a
stochastic innovation to wealth, this approach is consistent with the standard
Brandt-Santa-Clara methodology, where moments are estimated based on paths
generated under the current best-estimate of the optimal policy.


## Appendix

### Proof of the rewritten budget constraint
In this section we will show
```math
    W_{M + 1} = W_{n + 1} \left(\prod_{m = n + 1}^{M} G_{m + 1}\right)
    + \sum_{m = n + 1}^{M} Y_{m} \left(\prod_{p = m + 1}^{M} G_{p + 1}\right).
```
holds true for any time $n$, where $G_{m + 1} = (\omega_m^\top R^e_{m + 1} + X_m)$.


#### Base Case:
Let $M = n + 1$.
From the budget constraint we have $W_{n+2} = W_{n+1} G_{n+2} + Y_{n+1}$.
Using the formula for $M = n+1$:
The first term yields
```math
W_{n+1} \prod_{j=n+1}^{n+1} G_{j+1} = W_{n+1} G_{n+2},
```
while the second term yields
```math
\sum_{j=n+1}^{n+1} Y_j \prod_{k=j+1}^{n+1} G_{k+1} = Y_{n+1} \cdot (1) = Y_{n+1}.
```
where we used that the product $\prod_{k=n+2}^{n+1}$ is empty, thus equals 1.
Hence, the base case holds.


#### Inductive Step:
Assume the formula holds for some $M = K$.
That is:
```math
W_{K+1} = W_{n+1} \left( \prod_{j=n+1}^{K} G_{j+1} \right) + \sum_{j=n+1}^{K} Y_j \left( \prod_{k=j+1}^{K} G_{k+1} \right)
```
We examine the wealth at $M = K + 1$ (which is $W_{K+2}$) using the budget constraint:
```math
W_{K+2} = W_{K+1} G_{K+2} + Y_{K+1}
```
Substitute our inductive hypothesis for $W_{K+1}$

```math
    W_{K+2} = \left[ W_{n+1} \left( \prod_{j=n+1}^{K} G_{j+1} \right) + \sum_{j=n+1}^{K} Y_j \left( \prod_{k=j+1}^{K} G_{k+1} \right) \right] G_{K+2} + Y_{K+1}
```

Distribute $G_{K+2}$ into both terms
```math
W_{K+2} = W_{n+1} \left( \prod_{j=n+1}^{K} G_{j+1} \cdot G_{K+2} \right) + \sum_{j=n+1}^{K} Y_j \left( \prod_{k=j+1}^{K} G_{k+1} \cdot G_{K+2} \right) + Y_{K+1}
```
The first part merges into the product: $W_{n+1} \prod_{j=n+1}^{K+1} G_{j+1}$.
The $G_{K+2}$ term enters the summation's product, updating the upper bound to $K+1$.
The $Y_{K+1}$ term is equivalent to $Y_{K+1} \prod_{k=K+2}^{K+1} G_{k+1}$, which allows us to fold it into the summation as the $j = K+1$ term.
Combining these:
```math
W_{K+2} = W_{n+1} \left( \prod_{j=n+1}^{K+1} G_{j+1} \right) + \sum_{j=n+1}^{K+1} Y_j \left( \prod_{k=j+1}^{K+1} G_{k+1} \right)
```
The formula holds for $M = K+1$.

## References
```@bibliography
```