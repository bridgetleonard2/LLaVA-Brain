import numpy as np
from sklearn.preprocessing import StandardScaler
from himalaya.kernel_ridge import KernelRidgeCV

from himalaya.backend import set_backend
from sklearn.pipeline import make_pipeline


backend = set_backend("torch_cuda", on_error="warn")
print(backend)

scaler = StandardScaler(with_mean=True, with_std=False)

X_train = np.load(
    "results/features/clip_pipeline/multi-modal_projector/clip_85_features.npy"
    )
Y_train = np.load("data/clip/train_fmri/clip_85.npy")
X_test = np.load(
    "results/features/clip_pipeline/multi-modal_projector/clip_15_features.npy"
    )
Y_test = np.load("data/clip/test_fmri/clip_15.npy")

X_train = np.mean(X_train, axis=1)
X_test = np.mean(X_test, axis=1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

alphas = np.logspace(1, 20, 20)
cv = 7

kernel_ridge_cv = KernelRidgeCV(
    alphas=alphas, cv=cv,
    solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                       n_targets_batch_refit=100))

pipeline = make_pipeline(
    scaler,
    kernel_ridge_cv,
)

_ = pipeline.fit(X_train, Y_train)

coef = pipeline[-1].get_primal_coef()
coef = backend.to_numpy(coef)
print("n_features, n_voxels) =", coef.shape)
# # Get encoding model from coefficients
# # Regularize coefficients
# coef /= np.linalg.norm(coef, axis=0)[None]

# delayer = pipeline.named_steps['delayer']
# coef_per_delay = delayer.reshape_by_delays(coef, axis=0)
# print("(n_delays, n_features, n_voxels) =", coef_per_delay.shape)

# average_coef = np.mean(coef_per_delay, axis=0)
# print("(n_features, n_voxels) =", average_coef.shape)

X_test_scaled = (pipeline.named_steps['standardscaler'].transform(X_test))
Y_pred = np.matmul(X_test_scaled, coef)

# Caculate r2 between Y_pred and Y_test
residuals = Y_test - Y_pred
ss_res = np.sum(residuals ** 2, axis=0)
ss_tot = np.sum((Y_test - np.mean(Y_test, axis=0)) ** 2, axis=0)
r2 = 1 - ss_res / ss_tot
print("(n_voxels,) =", r2.shape)
print("Mean R2 score:", np.mean(r2))
print("Max R2 score:", np.max(r2))

scores = pipeline.score(X_test, Y_test)
print("(n_voxels,) =", scores.shape)

scores = backend.to_numpy(scores)
print("Mean R2 score:", np.mean(scores))
print("Max R2 score:", np.max(scores))
