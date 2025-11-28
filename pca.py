import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# EMBEDDED 16x16 IMAGE: Stylized digit "2"
# ---------------------------------------------------
X = np.array([
    [0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

print("="*60)
print("PCA Step-by-Step: Geometry, Decorrelation, Compression")
print("="*60)
print(f"Image shape: {X.shape}, Total values: {X.size}")

# ---------------------------------------------------
# TODO: CENTER THE DATA (subtract column-wise mean)
# ---------------------------------------------------
X_mean = np.mean(X, axis=0)  # shape (16,)
X_centered = X - X_mean  # shape (16, 16)

# Number of samples
n = X_centered.shape[0] if X_centered is not None else 16

# ---------------------------------------------------
# (a) GEOMETRIC TRANSFORMATION (SVD)
# ---------------------------------------------------
print("\n" + "-"*50)
print("(a) GEOMETRIC TRANSFORMATION")
print("-"*50)

# TODO: Compute SVD of X_centered: U, s, Vt = np.linalg.svd(...)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
V = Vt.T  # shape (16, 16)

# TODO: Rotate the data: 
Z = X_centered @ V  # shape (16, 16)

# Print first PC (only if computed)
if V is not None:
    print(f"First principal component (v1): {V[:, 0]}")

# Plot (only if data exists)
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray', vmin=0, vmax=255)
plt.title("Original")
plt.axis('off')

if Z is not None:
    plt.subplot(1, 2, 2)
    plt.imshow(Z, cmap='gray', aspect='auto')
    plt.title("Rotated Data (Z)")
    plt.xlabel("PC")
    plt.ylabel("Row")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# (b) DECORRELATION
# ---------------------------------------------------
print("\n" + "-"*50)
print("(b) STATISTICAL DECORRELATION")
print("-"*50)

# TODO: Compute covariance of original centered data
C_orig = (X_centered.T @ X_centered) / n  

# TODO: Compute covariance of rotated data (Z)
C_pca = (Z.T @ Z) / n  

# Compute off-diagonal energy
off_orig = np.sqrt(np.sum(C_orig**2) - np.sum(np.diag(C_orig)**2)) if C_orig is not None else 0
off_pca = np.sqrt(np.sum(C_pca**2) - np.sum(np.diag(C_pca)**2)) if C_pca is not None else 0

print(f"Off-diagonal energy: Original = {off_orig:.2f}, PCA = {off_pca:.2e}")

# Plot covariances
plt.figure(figsize=(8, 3))
if C_orig is not None:
    plt.subplot(1, 2, 1)
    plt.imshow(C_orig, cmap='coolwarm')
    plt.title("Cov (Original)")
if C_pca is not None:
    plt.subplot(1, 2, 2)
    plt.imshow(C_pca, cmap='coolwarm')
    plt.title("Cov (PCA)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# (c) COMPRESSION: SVD RECONSTRUCTION FOR MULTIPLE k
# ---------------------------------------------------
print("\n" + "-"*50)
print("(c) DIMENSIONALITY REDUCTION (SVD)")
print("-"*50)

k_values = [1, 3, 5, 10, 16]
reconstructions_svd = {}
errors_svd = {}

for k in k_values:
    # TODO: Reconstruct using top k components from SVD
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    X_rec = U_k @ np.diag(s_k) @ Vt_k + X_mean  # shape (16, 16)
    
    reconstructions_svd[k] = X_rec
    err = np.linalg.norm(X - X_rec, 'fro')
    errors_svd[k] = err
    print(f"k={k:2d} | Error = {err:6.2f}")

# Storage analysis
storage_k3 = 17*3 + 16
print(f"\nStorage: Original={X.size}, k=3 uses {storage_k3} → {X.size/storage_k3:.1f}x compression")

# Plot progressive reconstruction
plt.figure(figsize=(14, 3))
plt.subplot(1, len(k_values)+1, 1)
plt.imshow(X, cmap='gray', vmin=0, vmax=255)
plt.title("Original")
plt.axis('off')

for i, k in enumerate(k_values):
    plt.subplot(1, len(k_values)+1, i+2)
    img_to_show = reconstructions_svd[k] if reconstructions_svd[k] is not None else np.zeros_like(X)
    plt.imshow(img_to_show, cmap='gray', vmin=0, vmax=255)
    plt.title(f"k={k}\nErr={errors_svd[k]:.1f}")
    plt.axis('off')

plt.tight_layout()
plt.suptitle("SVD PCA: Reconstruction Quality vs. k", y=1.05)
plt.show()

# Variance curve
if s is not None:
    cumsum_var = np.cumsum(s**2) / np.sum(s**2)
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(s)+1), cumsum_var, 'bo-')
    plt.axhline(0.95, color='r', ls='--', label='95%')
    plt.xlabel("k"); plt.ylabel("Cumulative Variance"); plt.legend(); plt.grid(True)
    plt.title("Variance Explained")
    plt.show()

# ---------------------------------------------------
# (d) COVARIANCE METHOD (for comparison at k=3)
# ---------------------------------------------------
print("\n" + "-"*50)
print("(d) COVARIANCE METHOD vs SVD (at k=3)")
print("-"*50)

# TODO: Compute covariance matrix C 
C = (X_centered.T @ X_centered) / n

# TODO: Compute eigendecomposition of C (use np.linalg.eigh)
eigenvals, eigenvecs = np.linalg.eigh(C)

# Sort eigenvalues in descending order
if eigenvals is not None:
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

k = 3
if eigenvecs is not None:
    V_cov = eigenvecs[:, :k]
    X_rec_cov = X_centered @ V_cov @ V_cov.T + X_mean

error_cov = np.linalg.norm(X - X_rec_cov, 'fro') if X_rec_cov is not None else np.nan
svd_rec_at_3 = reconstructions_svd.get(3, None)
diff = np.linalg.norm(svd_rec_at_3 - X_rec_cov, 'fro') if (svd_rec_at_3 is not None and X_rec_cov is not None) else np.nan

print(f"SVD error (k=3):      {errors_svd.get(3, np.nan):.6f}")
print(f"Covariance error:     {error_cov:.6f}")
print(f"Reconstruction diff:  {diff:.2e}")

# Plot comparison
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.imshow(X, cmap='gray', vmin=0, vmax=255)
plt.title("Original")
plt.axis('off')

if svd_rec_at_3 is not None:
    plt.subplot(1, 3, 2)
    plt.imshow(svd_rec_at_3, cmap='gray', vmin=0, vmax=255)
    plt.title("SVD (k=3)")
    plt.axis('off')

if X_rec_cov is not None:
    plt.subplot(1, 3, 3)
    plt.imshow(X_rec_cov, cmap='gray', vmin=0, vmax=255)
    plt.title("Covariance (k=3)")
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("CONCLUSION & ANALYSIS")
print("="*60)
print("\n(a) GEOMETRIC TRANSFORMATION:")
print(f"    • X shape: {X.shape}")
print(f"    • X̃ (centered) shape: {X_centered.shape}")
print(f"    • Z (rotated) shape: {Z.shape}")
print(f"    • First PC (v1) captures structure in columns 4-9")
print(f"    • Decorrelation achieved: off-diagonal energy {off_orig:.2f} → {off_pca:.2e}")

print("\n(b) DECORRELATION POWER:")
print(f"    • Original covariance has off-diagonal energy: {off_orig:.2f}")
print(f"    • PCA covariance is diagonal with energy: {off_pca:.2e}")
print(f"    • Reduction factor: {off_orig/max(off_pca, 1e-15):.0e}x")

print("\n(c) COMPRESSION EFFICIENCY:")
print(f"    • Original: 256 values (16×16)")
print(f"    • k=3 storage: 16×3 (U_k) + 3 (σ) + 3×16 (V^T_k) + 16 (mean) = 67 values")
print(f"    • Compression ratio: {X.size/storage_k3:.1f}x")
print(f"    • Reconstruction error at k=3: {errors_svd[3]:.2f}")
print(f"    • Reconstruction error at k=5: {errors_svd[5]:.2f} (nearly perfect)")

print("\n(d) SVD vs COVARIANCE METHOD:")
print(f"    • SVD reconstruction error (k=3):    {errors_svd.get(3, np.nan):.6f}")
print(f"    • Covariance reconstruction error:   {error_cov:.6f}")
print(f"    • Difference in reconstructions:     {diff:.2e}")
print(f"    • Methods are numerically equivalent (diff ≈ machine epsilon)")

# (d) WHY SVD IS NUMERICALLY SUPERIOR:
#  Direct on X: SVD doesn't form X^T X (avoids squaring condition number
# Stability: κ(SVD) = σ_1/σ_n, κ(Cov) = (σ_1/σ_n)²
# Wide data: Works well even when n << d (wide matrix)
#  Ill-conditioned: SVD handles near-singular matrices better
#  No centering drift: SVD uses full precision of X_centered