import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

# Data preperation
df = pd.read_csv('mlm.csv')
df.head()
X = df[['x', 'y']]
Y = df['z']

# Prepare data for visualization
x = df['x']
y = df['y']
z = Y

# Visualize data
def set_ax(ax):
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    ax.locator_params(nbins=5, axis='x')
    ax.locator_params(nbins=5, axis='y')

def set_fig(fig, title, elev1, azim1, elev2, azim2, elev3, azim3):
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    axes = [ax1, ax2, ax3]
    ax1.view_init(elev=elev1, azim=azim1)
    ax2.view_init(elev=elev2, azim=azim2)
    ax3.view_init(elev=elev3, azim=azim3)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    return axes

plt.style.use('default')
fig = plt.figure(figsize=(12, 4))
axes = set_fig(fig, 'scatter plot', 50, 20, 28, 10, 70, 60)
for ax in axes:
    ax.scatter(x, y, z, c='k', marker='o', alpha=0.5)
    set_ax(ax)

# Linear regression model class using normal equation to fit
class LinearRegression(object):
    def __init__(self):
        self._theta = None
        self.intercept_ = None
        self.coef_ = None

    def predict(self, x):
        X_b = np.hstack([np.ones((len(x), 1)), x])
        return X_b.dot(self._theta)

    def fit(self, x, y):
        X_b = np.hstack([np.ones((len(x), 1)), x])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

# Train
m = LinearRegression()
m.fit(X, Y)
print(m.coef_)
predicted = m.predict(X)

# Use RMSE for evaluation
rmse = np.sqrt(((predicted - Y) ** 2).sum() / y.shape[0])
print(rmse)

# Prepare data for predicted hyperplane visualization
x_pred = np.linspace(-10, 110, 30)  # range of x
y_pred = np.linspace(-10, 110, 30)  # range of y
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
predicted = m.predict(model_viz)

# Plot data points and predicted hyperplane
fig2 = plt.figure(figsize=(12, 4))
axes = set_fig(fig2, '$rmse = %.2f$' %rmse, 36, 20, 17, 10, 70, 60)
for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5, markersize=4)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
    set_ax(ax)
plt.show()

# Plot data points and predicted hyperplane to gif
fig3 = plt.figure(figsize=(4, 4))
fig3.tight_layout()
fig3.suptitle('$rmse = %.2f$' %rmse, fontsize=20)
ax = fig3.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5, markersize=4)
ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
set_ax(ax)
for ii in np.arange(0, 360, 1):
    ax.view_init(elev=17, azim=ii)
    fig3.savefig('gif/gif_image%03d.png' %ii)

# filepaths
fp_in = "gif/gif_image*.png"
fp_out = "image.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=25, loop=0)

