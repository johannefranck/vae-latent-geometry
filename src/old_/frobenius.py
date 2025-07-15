import json
import numpy as np

# Load the data files
with open('src/artifacts/geodesic_distances_seed12_p133.json', 'r') as f:
    data_seed12 = json.load(f)

with open('src/artifacts/geodesic_distances_seed123_p133.json', 'r') as f:
    data_seed123 = json.load(f)

# Get cluster IDs
clusters_12 = data_seed12['cluster_ids']
clusters_123 = data_seed123['cluster_ids']

# Find differences between cluster sets
set_12 = set(clusters_12)
set_123 = set(clusters_123)

# Types excluded from seed123 but in seed12
excluded_from_123 = set_12 - set_123
# Types excluded from seed12 but in seed123
excluded_from_12 = set_123 - set_12
# Common types
common_types = set_12 & set_123

print("Types excluded from seed123 but present in seed12:")
for cluster_type in sorted(excluded_from_123):
    print(f"  - {cluster_type}")

print("\nTypes excluded from seed12 but present in seed123:")
for cluster_type in sorted(excluded_from_12):
    print(f"  - {cluster_type}")

print(f"\nTotal excluded from seed123: {len(excluded_from_123)}")
print(f"Total excluded from seed12: {len(excluded_from_12)}")
print(f"Common types: {len(common_types)}")

# Find indices of common types in both datasets
common_indices_12 = [i for i, cluster in enumerate(clusters_12) if cluster in common_types]
common_indices_123 = [i for i, cluster in enumerate(clusters_123) if cluster in common_types]

print(f"\nFiltering to {len(common_types)} common classes...")

# Get the distance matrices
matrix_12 = np.array(data_seed12['distance_matrix'])
matrix_123 = np.array(data_seed123['distance_matrix'])

# Filter matrices to only include common types
filtered_matrix_12 = matrix_12[np.ix_(common_indices_12, common_indices_12)]
filtered_matrix_123 = matrix_123[np.ix_(common_indices_123, common_indices_123)]

print(f"Original matrix shapes: {matrix_12.shape}, {matrix_123.shape}")
print(f"Filtered matrix shapes: {filtered_matrix_12.shape}, {filtered_matrix_123.shape}")

# Calculate Frobenius norms for each matrix separately
frobenius_12 = np.linalg.norm(filtered_matrix_12, 'fro')
frobenius_123 = np.linalg.norm(filtered_matrix_123, 'fro')

print(f"\nFrobenius norms:")
print(f"Seed 12 matrix: {frobenius_12:.4f}")
print(f"Seed 123 matrix: {frobenius_123:.4f}")

# Calculate the difference between the two matrices
matrix_difference = filtered_matrix_12 - filtered_matrix_123