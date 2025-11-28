import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------
np.random.seed(42)
M = 50
x = np.linspace(-3, 3, M)
y_true = 2 * np.sin(2 * x) + 0.5 * x**3
y_obs = y_true + np.random.normal(0, 0.5, M)
y = y_obs.reshape(-1, 1)

lambda_reg = 0.1
gamma = 0.5

print("="*60)
print("REGRESSION COMPARISON: Explicit vs. Kernel Methods")
print("="*60)
print(f"True function: y = 2*sin(2x) + 0.5*x^3")
print(f"Samples: {M}, Noise std: 0.5")

# ---------------------------------------------------
# (a) OLS AND POLYNOMIAL REGRESSION
# ---------------------------------------------------
# Linear features [1, x]
Phi_lin = np.vstack([np.ones_like(x), x]).T
w_lin = np.linalg.lstsq(Phi_lin, y, rcond=None)[0]
y_ols_pred = Phi_lin @ w_lin

# Polynomial features [1, x, ..., x^5]
Phi_poly = np.vander(x, N=6, increasing=True)
w_poly = np.linalg.lstsq(Phi_poly, y, rcond=None)[0]
y_poly_pred = Phi_poly @ w_poly

# ---------------------------------------------------
# (b) GRAMIAN RIDGE (LINEAR FEATURES)
# ---------------------------------------------------
K_lin = Phi_lin @ Phi_lin.T
alpha_lin = np.linalg.solve(K_lin + lambda_reg * np.eye(M), y)
y_gram_lin_pred = K_lin @ alpha_lin

# ---------------------------------------------------
# (c) RBF KERNEL RIDGE
# ---------------------------------------------------
# Compute RBF kernel matrix: K_ij = exp(-gamma * ||x_i - x_j||^2)
X1, X2 = np.meshgrid(x, x)
K_rbf = np.exp(-gamma * (X1 - X2)**2)
alpha_rbf = np.linalg.solve(K_rbf + lambda_reg * np.eye(M), y)
y_rbf_pred = K_rbf @ alpha_rbf

# ---------------------------------------------------
# (d) MSE ANALYSIS — STUDENTS MUST COMPLETE PREDICTIONS FIRST
# ---------------------------------------------------
# Compute MSE for each method (against true function)
mse_ols = mean_squared_error(y_true, y_ols_pred) if y_ols_pred is not None else np.nan
mse_poly = mean_squared_error(y_true, y_poly_pred) if y_poly_pred is not None else np.nan
mse_gram_lin = mean_squared_error(y_true, y_gram_lin_pred) if y_gram_lin_pred is not None else np.nan
mse_rbf = mean_squared_error(y_true, y_rbf_pred) if y_rbf_pred is not None else np.nan

# Print detailed MSE report
print("\n" + "="*60)
print("MEAN SQUARED ERROR (MSE) vs. TRUE FUNCTION")
print("="*60)
print(f"{'Method':<25} {'MSE':<12} {'Relative to Linear'}")
print("-"*60)
print(f"{'Linear OLS':<25} {mse_ols:<12.4e} {'(baseline)':<20}")
if not np.isnan(mse_gram_lin):
    ratio = mse_gram_lin / mse_ols if mse_ols > 0 else np.nan
    print(f"{'Gramian (Linear)':<25} {mse_gram_lin:<12.4e} {'(should be ~1.0)':<20}")
if not np.isnan(mse_poly):
    ratio = mse_ols / mse_poly if mse_poly > 0 else np.nan
    print(f"{'Polynomial (deg 5)':<25} {mse_poly:<12.4e} {f'({ratio:.1f}× better)':<20}")
if not np.isnan(mse_rbf):
    ratio = mse_ols / mse_rbf if mse_rbf > 0 else np.nan
    print(f"{'RBF Kernel':<25} {mse_rbf:<12.4e} {f'({ratio:.1f}× better)':<20}")

# Identify best method
methods = {
    'Linear OLS': mse_ols,
    'Gramian (Linear)': mse_gram_lin,
    'Polynomial (deg 5)': mse_poly,
    'RBF Kernel': mse_rbf
}
best_method = min(methods, key=methods.get)
print(f"\nBest method: {best_method} (MSE = {methods[best_method]:.4e})")

# ---------------------------------------------------
# PLOTTING — NOW WITH CLEAR VISUALIZATION OF GRAMIAN LINEAR
# ---------------------------------------------------
x_fine = np.linspace(-3, 3, 200)
y_true_fine = 2 * np.sin(2 * x_fine) + 0.5 * x_fine**3

# Prepare predictions
Phi_lin_fine = np.vstack([np.ones_like(x_fine), x_fine]).T
y_ols_fine = (Phi_lin_fine @ w_lin).flatten() if w_lin is not None else np.zeros_like(x_fine)

# Gramian (Linear) — same as OLS, but plot differently
y_gram_lin_fine = y_ols_fine  # identical prediction

# Polynomial
Phi_poly_fine = np.vander(x_fine, N=6, increasing=True)
y_poly_fine = (Phi_poly_fine @ w_poly).flatten() if w_poly is not None else np.zeros_like(x_fine)

# RBF Kernel
X1, X2 = np.meshgrid(x, x_fine)
K_rbf_test = np.exp(-gamma * (X1 - X2)**2)  # shape: (50, 200)
y_rbf_fine = (K_rbf_test @ alpha_rbf).flatten() if alpha_rbf is not None else np.zeros_like(x_fine)

# Plot
plt.figure(figsize=(12, 7))
plt.scatter(x, y_obs, color='gray', alpha=0.6, label='Observed Data')
plt.plot(x_fine, y_true_fine, 'k--', linewidth=2, label='True Function')

# Linear OLS — solid red
plt.plot(x_fine, y_ols_fine, 'r-', linewidth=2, label=f'Linear OLS (MSE={mse_ols:.1e})')

# Gramian (Linear) — use orange dotted with markers to make visible
plt.plot(x_fine, y_gram_lin_fine, 'orange', linestyle=':', linewidth=2, 
         marker='.', markevery=20, label=f'Gramian (Linear) (MSE={mse_gram_lin:.1e})')

# Polynomial — green dashed
plt.plot(x_fine, y_poly_fine, 'g--', linewidth=2, label=f'Poly Deg 5 (MSE={mse_poly:.1e})')

# RBF Kernel — blue solid
plt.plot(x_fine, y_rbf_fine, 'b-', linewidth=2, label=f'RBF Kernel (MSE={mse_rbf:.1e})')

# Optional: Annotate Gramian line to ensure visibility
mid_x = 0
mid_y = np.interp(mid_x, x_fine, y_gram_lin_fine)
plt.annotate('Gramian (Linear)', xy=(mid_x, mid_y), xytext=(mid_x+0.5, mid_y-2),
             arrowprops=dict(facecolor='orange', shrink=0.05),
             fontsize=9, color='orange')

plt.title('Regression Methods Compared (Gramian Linear Now Visible)')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# TODO: ANALYSIS
"""
1. Why is MSE of Gramian (Linear) nearly identical to Linear OLS?
2. How much better is RBF than polynomial? Why?
3. Could we use Gram matrix for RBF without knowing φ(x)? What does this enable?
"""