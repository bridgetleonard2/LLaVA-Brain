import numpy as np
from sklearn.preprocessing import StandardScaler
from delayerClass import Delayer
from himalaya.kernel_ridge import KernelRidgeCV

from himalaya.backend import set_backend
from sklearn.pipeline import make_pipeline


backend = set_backend("torch_cuda", on_error="warn")
print(backend)

scaler = StandardScaler(with_mean=True, with_std=False)
delayer = Delayer(delays=[1, 2, 3, 4])

data_dir = 'data/clip'

X_train = np.load(f"{data_dir}/train_stim/clip_85.npy")
Y_train = np.load(f"{data_dir}/train_fmri/clip_85.npy")
X_test = np.load(f"{data_dir}/test_stim/clip_15.npy")
Y_test = np.load(f"{data_dir}/test_fmri/clip_15.npy")

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
    delayer,
    kernel_ridge_cv,
)

_ = pipeline.fit(X_train, Y_train)
scores = pipeline.score(X_test, Y_test)
print("(n_voxels,) =", scores.shape)

print("Mean R2 score:", np.mean(scores))
print("Max R2 score:", np.max(scores))
