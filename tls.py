import numpy as np
import matplotlib.pyplot as plt

def compute_ols(A, y):
    """Compute OLS estimate for y = theta * x (A is (M,1))"""
    # Ordinary Least Squares: theta = (A^T A)^{-1} A^T y
    # A has shape (M,1), y has shape (M,1) -> result is a (1,1) array
    # Return scalar
    AtA = A.T @ A
    Aty = A.T @ y
    # Use np.linalg.solve for numerical stability when possible
    try:
        theta = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(AtA) @ Aty
    return float(theta.squeeze())

def compute_tls(A, y):
    """Compute TLS estimate from augmented matrix [A | y]"""
    # Stack A and y into an (M,2) matrix and compute SVD
    B = np.hstack([A, y])
    # SVD: B = U S Vt; last column of V (or last row of Vt) is the smallest singular vector
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    v = Vt.T[:, -1]  # shape (2,)
    # For the model y = theta * x, the TLS solution satisfies [x y] [a; b] = 0
    # with parameter vector proportional to [a; b] = [theta; -1]. Thus theta = -a/b
    if abs(v[1]) < 1e-12:
        # degenerate case; fall back to OLS
        return compute_ols(A, y)
    theta_tls = -v[0] / v[1]
    return float(theta_tls)

# -----------------------------
# COMMON SETUP
# -----------------------------
np.random.seed(42)
theta_true = 2.5
M = 50
x_clean = np.linspace(1, 10, M)
y_clean = theta_true * x_clean
noise_std = 1.2  # Use same noise level in both experiments

# =====================================================
# EXPERIMENT 1: ONLY y IS NOISY (x is clean)
# =====================================================
print("=" * 50)
print("EXPERIMENT 1: Noise only in y")
print("=" * 50)

x_obs1 = x_clean.copy()                          # x is clean
y_obs1 = y_clean + np.random.normal(0, noise_std, M)

A1 = x_obs1.reshape(-1, 1)
y1 = y_obs1.reshape(-1, 1)

theta_OLS1 = compute_ols(A1, y1)
theta_TLS1 = compute_tls(A1, y1)

err_OLS1 = abs(theta_OLS1 - theta_true)
err_TLS1 = abs(theta_TLS1 - theta_true)

print(f"True theta: {theta_true:.4f}")
print(f"OLS estimate: {theta_OLS1:.4f} (error = {err_OLS1:.4f})")
print(f"TLS estimate: {theta_TLS1:.4f} (error = {err_TLS1:.4f})")

# Plot Exp 1
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_obs1, y_obs1, alpha=0.6, label='Data (y noisy)')
plt.plot(x_clean, theta_true * x_clean, 'k--', label='True line')
plt.plot(x_clean, theta_OLS1 * x_clean, 'r-', label=f'OLS (err={err_OLS1:.3f})')
plt.plot(x_clean, theta_TLS1 * x_clean, 'b-', label=f'TLS (err={err_TLS1:.3f})')
plt.title('Exp 1: Noise only in y')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(True, linestyle=':', alpha=0.7)

# =====================================================
# EXPERIMENT 2: BOTH x AND y ARE NOISY
# =====================================================
print("\n" + "=" * 50)
print("EXPERIMENT 2: Noise in both x and y")
print("=" * 50)

x_obs2 = x_clean + np.random.normal(0, noise_std, M)  # x is noisy!
y_obs2 = y_clean + np.random.normal(0, noise_std, M)  # y is noisy!

A2 = x_obs2.reshape(-1, 1)
y2 = y_obs2.reshape(-1, 1)

theta_OLS2 = compute_ols(A2, y2)
theta_TLS2 = compute_tls(A2, y2)

err_OLS2 = abs(theta_OLS2 - theta_true)
err_TLS2 = abs(theta_TLS2 - theta_true)

print(f"True theta: {theta_true:.4f}")
print(f"OLS estimate: {theta_OLS2:.4f} (error = {err_OLS2:.4f})")
print(f"TLS estimate: {theta_TLS2:.4f} (error = {err_TLS2:.4f})")

# Plot Exp 2
plt.subplot(1, 2, 2)
plt.scatter(x_obs2, y_obs2, alpha=0.6, label='Data (x and y noisy)')
plt.plot(x_clean, theta_true * x_clean, 'k--', label='True line')
plt.plot(x_clean, theta_OLS2 * x_clean, 'r-', label=f'OLS (err={err_OLS2:.3f})')
plt.plot(x_clean, theta_TLS2 * x_clean, 'b-', label=f'TLS (err={err_TLS2:.3f})')
plt.title('Exp 2: Noise in x and y')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()

# =====================================================
# TODO: ANALYSIS (answer in comments below)
# =====================================================
"""
After running both experiments, answer:

1. In Experiment 1, which method was more accurate? Why?
2. In Experiment 2, which method was more accurate? Why?
3. What assumption does OLS make about the input x?
4. When should you prefer TLS over OLS?
"""

"""Analysis answers:

1) Experiment 1 (only y noisy): OLS is more accurate.
    Reason: OLS assumes the predictor `x` is measured without error and minimizes vertical
    distances (errors in y). When x is clean and only y is noisy, OLS is the optimal
    unbiased estimator (in the classical linear model). TLS, which assumes errors in
    both variables and minimizes orthogonal distances, will unnecessarily rotate the fit
    and typically introduce bias when x has no noise.

2) Experiment 2 (both x and y noisy): TLS is more accurate.
    Reason: When both x and y contain noise of comparable magnitude, the errors-in-variables
    model is appropriate. TLS accounts for noise in both variables by minimizing orthogonal
    distances to the fitted line, reducing the attenuation/bias that OLS suffers from when
    regressors are noisy.

3) OLS assumption about x:
    OLS assumes that the input/predictor `x` is measured without error (or that the
    regressors are fixed/non-random or independent of the noise). Violating this leads to
    biased estimates (attenuation bias for slopes).

4) When to prefer TLS over OLS:
    Prefer TLS when the independent variables (x) are also corrupted by measurement noise
    and errors in x and y are of comparable scale, i.e., an errors-in-variables situation.
    Do not use TLS when regressors are effectively noise-free — OLS is preferable there.
"""