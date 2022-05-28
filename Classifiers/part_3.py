import SoftSVM
import matplotlib.pyplot as plt
import seaborn as sns
import verify_gradients
import numpy as np

#Creating moon dataset
from sklearn.datasets import make_moons
X_moons, y_moons = make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=156)
y_moons = ((2 * y_moons) - 1)[:, None] 
"""print(f"{X_moons.shape}, {y_moons.shape}") 
plt.figure(), plt.grid(alpha=0.5), plt.title("Synthetic moon dataset") 
_ = sns.scatterplot(x=X_moons[:, 0], y=X_moons[:, 1], hue=y_moons[:, 0])
"""

verify_gradients.compare_gradients(X_moons, y_moons, deltas=np.logspace(-9, -1, 12))