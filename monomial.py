import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# STEP 1: SAMPLE THE FUNCTION
# ---------------------------------------------------
N = 50
t = np.linspace(0, 1, N)
y = np.exp(t)  # f(t) = e^t

# ---------------------------------------------------
# TODO (a): BUILD DESIGN MATRIX A, GRAM MATRIX G, AND RHS b
# ---------------------------------------------------
A = np.column_stack([np.ones(N), t, t**2])  # shape (N, 3), columns [1, t, t^2]
G = A.T @ A  # A^T A
b = A.T @ y  # A^T y

a_LS = np.linalg.solve(G, b)  # solve G a = b
kappa_G = np.linalg.cond(G)  # condition number of G

print("--- (a) LS Solution ---")
print(f"a_LS = {a_LS}")
print(f"Condition number κ(G) = {kappa_G:.2e}")

# ---------------------------------------------------
# TODO (b): IMPLEMENT TRUNCATED SVD FOR R' = 2
# ---------------------------------------------------
U, s, Vt = np.linalg.svd(A, full_matrices=False)  # compute SVD of A
V = Vt.T
R_prime = 2

# Build truncated pseudo-inverse using top R_prime singular values
A_dag_trunc = V[:, :R_prime] @ np.diag(1.0 / s[:R_prime]) @ U[:, :R_prime].T
a_TSVD = A_dag_trunc @ y

print("\n--- (b) TSVD Solution (R'=2) ---")
print(f"Singular values of A: {s}")
print(f"a_TSVD = {a_TSVD}")

# ---------------------------------------------------
# TODO (c): NORMS AND PLOTTING
# ---------------------------------------------------
norm_LS = np.linalg.norm(a_LS)      # ||a_LS||_2
norm_TSVD = np.linalg.norm(a_TSVD)    # ||a_TSVD||_2

print("\n--- (c) Coefficient Norms ---")
print(f"||a_LS||_2 = {norm_LS:.4f}")
print(f"||a_TSVD||_2 = {norm_TSVD:.4f}")

# Evaluate approximations
t_fine = np.linspace(0, 1, 200)
p_LS = a_LS[0] + a_LS[1]*t_fine + a_LS[2]*t_fine**2
p_TSVD = a_TSVD[0] + a_TSVD[1]*t_fine + a_TSVD[2]*t_fine**2

plt.figure(figsize=(10, 6))
plt.plot(t_fine, np.exp(t_fine), 'k--', label=r'$f(t) = e^t$')
plt.plot(t_fine, p_LS, 'r-', label='LS Solution')
plt.plot(t_fine, p_TSVD, 'b-', label='TSVD Solution (R\'=2)')
plt.scatter(t, y, color='gray', s=10, alpha=0.5, label='Sampled data')
plt.title(f"Approximation (κ(G) = {kappa_G:.2e})")
plt.xlabel('t'); plt.ylabel('f(t)')
plt.legend(); plt.grid(True)
plt.show()

# ---------------------------------------------------
# TODO (d): ERROR ANALYSIS
# ---------------------------------------------------
# Compute residual norms: ||A a - y||_2
res_LS = np.linalg.norm(A @ a_LS - y)      # LS residual
res_TSVD = np.linalg.norm(A @ a_TSVD - y)    # TSVD residual

print("\n--- (d) Residual Errors ---")
print(f"LS residual      = {res_LS:.6f}")
print(f"TSVD residual    = {res_TSVD:.6f}")

# Optional: Sweep over all possible truncation levels (1 to 3)
R_vals = [1, 2, 3]
residuals = []
coeff_norms = []

for R in R_vals:
    # TODO: Compute a_R using truncated SVD with R components
    # Reuse U, s, V from above
    if R <= len(s):
        Sigma_inv_R = np.diag(1.0 / s[:R])
        A_dag_R = V[:, :R] @ Sigma_inv_R @ U[:, :R].T
        a_R = A_dag_R @ y
        res_R = np.linalg.norm(A @ a_R - y)
        residuals.append(res_R)
        coeff_norms.append(np.linalg.norm(a_R))
    else:
        residuals.append(np.nan)
        coeff_norms.append(np.nan)

# Plot residual vs R'
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(R_vals, residuals, 'o-', color='purple')
plt.title('Residual Norm vs Truncation Level')
plt.xlabel("R' (number of singular values retained)")
plt.ylabel(r"$\|A \mathbf{a} - \mathbf{y}\|_2$")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(R_vals, coeff_norms, 's-', color='green')
plt.title('Coefficient Norm vs Truncation Level')
plt.xlabel("R'")
plt.ylabel(r"$\|\mathbf{a}\|_2$")
plt.grid(True)

plt.tight_layout()
plt.show()

# EXPLANATION OF TRADE-OFFS:
# 
# Why does residual increase as R' decreases?
# - The truncated SVD A†_R' excludes the contribution of small singular values,
#   which contain information about fitting higher-frequency components of the data.
#   When we use fewer singular values (smaller R'), we lose representational power,
#   so the residual ||Aa_R' - y||_2 increases. The truncated solution cannot fit
#   the data as well.
#
# Why does coefficient norm decrease?
# - Smaller singular values amplify noise when inverted (1/σ_i becomes very large).
#   By discarding them, we avoid using these amplified "noisy" directions.
#   The coefficient norm ||a_R'||_2 is directly related to the energy in these
#   directions, so smaller R' means lower coefficient norms.
#
# What does this say about overfitting vs regularization?
# - Full least squares (R'=3) fits all data including noise, leading to large
#   coefficient norms and potential overfitting. Truncated SVD (R'<3) acts as
#   implicit regularization by filtering out ill-conditioned directions.
# - The trade-off: accepting slightly higher residuals (less perfect fit) in
#   exchange for lower coefficient norms and better generalization/stability.
# - This is the bias-variance trade-off: regularization reduces variance (stability)
#   at the cost of some bias (residual error).