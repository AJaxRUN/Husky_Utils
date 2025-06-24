import yaml
import numpy as np
import pybullet as p
import pybullet_data
import time

def generate_grid(center, x_count, y_count, z_count, spacing):
    grid = []
    for i in range(x_count):
        for j in range(y_count):
            for k in range(z_count):
                point = [
                    center[0] + (i - (x_count - 1) / 2) * spacing[0],
                    center[1] + (j - (y_count - 1) / 2) * spacing[1],
                    center[2] + (k - (z_count - 1) / 2) * spacing[2],
                ]
                grid.append(point)
    return grid

def add_spheres_to_link(link_name, centers, radius):
    if isinstance(centers[0], list):  # multiple centers
        return [{ "center": centers, "radius": radius }]
    else:  # single center
        return [{ "center": centers, "radius": radius }]

def save_yaml(sphere_dict, file_path='husky_spheres.yaml'):
    with open(file_path, 'w') as f:
        yaml.dump(sphere_dict, f, sort_keys=False)
    print(f"YAML saved to: {file_path}")

def visualize_husky_and_spheres(sphere_data, urdf_path='husky.urdf'):
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.loadURDF(urdf_path, [0, 0, 0.1])

    for link_name, spheres in sphere_data.items():
        for entry in spheres:
            centers = entry["center"]
            if not isinstance(centers[0], list):
                centers = [centers]
            for c in centers:
                vis_id = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=entry["radius"],
                    rgbaColor=[1, 0, 0, 0.6]
                )
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=vis_id,
                    basePosition=c
                )

    print("Press Ctrl+C to exit visualization.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        p.disconnect()

# === USER CONFIGURATION ===
sphere_data = {}

# Add base_link spheres in 2 layers
sphere_data["base_link"] = add_spheres_to_link(
    "base_link",
    # centers=[[0.0, 0.0, 0.265], [0.0, 0.0, 0.125]],
    centers=[[0.0, 0.0, 0.0]],
    radius=0.3
)

# Add grid
# chassis_grid = generate_grid(center=[0.0, 0.0, 0.2], x_count=5, y_count=4, z_count=2, spacing=[0.2, 0.2, 0.2])
# sphere_data["chassis_link"] = add_spheres_to_link("chassis_link", chassis_grid, radius=0.15)

# === EXPORT YAML ===
save_yaml(sphere_data, "husky_spheres.yaml")

# === VISUALIZE ===
visualize_husky_and_spheres(sphere_data, urdf_path="husky.urdf")
