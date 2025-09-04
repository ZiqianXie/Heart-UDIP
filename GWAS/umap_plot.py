import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import umap
import seaborn as sns
import mplcursors

# Load the 256D feature CSV file
input_csv = ".../UKB_60000.csv"
df = pd.read_csv(input_csv)

# Extract Patient IDs and features
patient_ids = df.iloc[:, 0].values  # First column assumed to be Patient_ID
features = df.iloc[:, 2:].values    # Features start from third column

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
features_umap = umap_model.fit_transform(features_scaled)

# Compute density
def get_density(x, y):
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    return density

umap_density = get_density(features_umap[:, 0], features_umap[:, 1])

# Plot with mplcursors for hoverable Patient_IDs
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(features_umap[:, 0], features_umap[:, 1], 
                     c=umap_density, cmap='Spectral', alpha=0.6, s=2)

plt.colorbar(scatter, label="Density")
plt.title("UMAP Visualization of 4ch-view 256D Features")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

# Add interactive cursor
cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_add(sel):
    index = sel.index
    sel.annotation.set(text=f"Patient_ID: {patient_ids[index]}")
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

plt.tight_layout()
plt.show()


