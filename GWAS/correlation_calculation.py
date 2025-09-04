import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Read CSV
input_csv = ".../UKB_60000.csv"
df = pd.read_csv(input_csv)

# 2. Extract feature matrix (assuming 3rd column onward)
features = df.iloc[:, 2:]  # Shape: (n_samples, 256)

# 3. Compute correlation matrix
correlation_matrix = features.corr(method='pearson')
"""
print(np.mean(abs(correlation_matrix)))
def mean_abs_corr(corr_matrix):
    abs_corr = corr_matrix.abs()
    upper_triangle = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
    return upper_triangle.stack().mean()
print(mean_abs_corr(correlation_matrix))
"""
# 4. Calculate mean and standard deviation of absolute correlations
def mean_sd_abs_corr(corr_matrix):
    abs_corr = corr_matrix.abs()
    upper_triangle = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
    vals = upper_triangle.stack().values
    mean_val = vals.mean()
    std_val = vals.std()
    return mean_val, std_val

mean_corr, std_corr = mean_sd_abs_corr(correlation_matrix)
print(f"Mean absolute correlation: {mean_corr:.4f}")
print(f"Standard deviation of absolute correlation: {std_corr:.4f}")

# 5. Plot the correlation matrix
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True, 
            xticklabels=50, yticklabels=50, cbar_kws={"shrink": 0.5})
plt.title("Correlation Matrix of 2Ch_view Features")
plt.xlabel("Feature Index")
plt.ylabel("Feature Index")
plt.tight_layout()
plt.show()
