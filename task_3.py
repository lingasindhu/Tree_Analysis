import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load dataset using Dask
file_path = "GUY01_000.txt"
df = dd.read_csv(file_path, delimiter=" ", header=None)
points = df.compute().values[:, :3]  # Convert to NumPy array (X, Y, Z coordinates)

# Define parameters for cylinder fitting
bin_size = 5  # Height interval for cylinder segments
radius_threshold = 1.7  # Max allowed radius change before crown starts
angle_threshold = 2  # Max allowed deviation angle before crown starts

# Step 1: Divide the tree into vertical bins
z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
bins = np.arange(z_min, z_max + bin_size, bin_size)

cylinder_info = []  # Store radius, deviation, and classification info

for i in range(len(bins) - 1):
    bin_start, bin_end = bins[i], bins[i + 1]
    points_in_bin = points[(points[:, 2] >= bin_start) & (points[:, 2] < bin_end)]
    
    if len(points_in_bin) < 10:  # Skip bins with too few points
        continue
    
    # Step 2: Perform PCA to determine orientation
    pca = PCA(n_components=2)
    pca.fit(points_in_bin[:, :2])  # Fit X-Y coordinates
    dominant_axis = pca.components_[0]  # Primary tree direction
    deviation_angle = np.arccos(np.dot(dominant_axis, [1, 0])) * 180 / np.pi  # Convert to degrees
    
    # Compute radius as avg. distance from PCA center
    mean_x, mean_y = np.mean(points_in_bin[:, :2], axis=0)
    radius = np.mean(np.sqrt((points_in_bin[:, 0] - mean_x)**2 + (points_in_bin[:, 1] - mean_y)**2))
    
    cylinder_info.append((i + 1, bin_start, radius, deviation_angle))

# Step 3: Detect Crown Transition
crown_start_z = None
for i in range(1, len(cylinder_info)):
    cyl_num_prev, z_prev, radius_prev, angle_prev = cylinder_info[i - 1]
    cyl_num_curr, z_curr, radius_curr, angle_curr = cylinder_info[i]
    
    radius_diff = abs(radius_curr - radius_prev)
    angle_diff = abs(angle_curr - angle_prev)

    # Identify crown transition OR reclassify as stem if deviation/radius match threshold
    if radius_diff > radius_threshold and angle_diff > angle_threshold:
        crown_start_z = z_curr
        print(f"🔹 Crown starts at Bin {cyl_num_curr}, Height: {z_curr:.2f} cm")
        break  # Stop at first significant transition

# Step 4: Print all cylinder/bin details
print("\nCylinder Information:")
print("Cylinder No | Bin Height (cm) | Radius (cm) | Deviation Angle (°)")
print("-" * 50)

for cyl_num, z_bin, radius, deviation in cylinder_info:
    bin_classification = "STEM" if z_bin < crown_start_z else "CROWN"
    print(f"{cyl_num:^12} | {z_bin:^15.2f} | {radius:^12.2f} | {deviation:^14.2f} → {bin_classification}")

# Step 5: Save segmented points to separate files
stem_points = points[points[:, 2] < crown_start_z]
crown_points = points[points[:, 2] >= crown_start_z]

np.savetxt("stem_points.txt", stem_points, fmt="%.4f", comments="")
np.savetxt("crown_points.txt", crown_points, fmt="%.4f", comments="")

print("\n✅ Stem points saved to 'stem_points.txt'")
print("✅ Crown points saved to 'crown_points.txt'")

# Step 6: 3D Visualization with embedded bin info
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection="3d")

# Color bins and add text labels
colors = plt.cm.viridis(np.linspace(0, 1, len(bins) - 1))
for i, (cyl_num, z_bin, radius, deviation) in enumerate(cylinder_info):
    bin_points = points[(points[:, 2] >= z_bin) & (points[:, 2] < z_bin + bin_size)]
    ax.scatter(bin_points[:, 0], bin_points[:, 1], bin_points[:, 2], color=colors[i], s=10)
    
    # Label bins with cylinder number, radius, and deviation angle
    ax.text(np.mean(bin_points[:, 0]), np.mean(bin_points[:, 1]), z_bin, 
            f"Cyl {cyl_num}\nR:{radius:.2f}cm\nΔ:{deviation:.2f}°", 
            color="black", fontsize=8, ha="center")

# Highlight crown transition
ax.axhline(y=crown_start_z, color="red", linestyle="--", label=f"Crown Starts at {crown_start_z:.2f} cm")

ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Height (Z)")
ax.set_title("Tree Segmentation with Embedded Cylinder Information")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.show()