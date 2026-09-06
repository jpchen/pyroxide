"""Generate the shared synthetic datasets used by both benchmark harnesses."""
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# logistic regression: N x D design, labels from true coefs 1..D
N, D = 1000, 5
X = rng.standard_normal((N, D))
true = np.arange(1, D + 1, dtype=float)
p = 1 / (1 + np.exp(-(X @ true)))
y = (rng.random(N) < p).astype(float)
np.savetxt(os.path.join(here, "data", "logreg_X.csv"), X, delimiter=",", fmt="%.10f")
np.savetxt(os.path.join(here, "data", "logreg_y.csv"), y, delimiter=",", fmt="%.0f")
print("wrote logreg", X.shape)
