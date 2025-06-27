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
        return [{"center": centers, "radius": radius}]
    else:  # single center
        return [{"center": centers, "radius": radius}]

def save_yaml(sphere_dict, file_path='husky_spheres.yaml'):
    # Custom YAML Dumper to format lists and floats
    class CustomDumper(yaml.Dumper):
        def represent_float(self, data):
            return self.represent_scalar('tag:yaml.org,2002:float', f"{data:.3f}")
        
        def represent_list(self, data):
            if isinstance(data, list) and all(isinstance(item, (int, float)) for item in data):
                return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
            return super().represent_sequence('tag:yaml.org,2002:seq', data)

    CustomDumper.add_representer(float, CustomDumper.represent_float)
    CustomDumper.add_representer(list, CustomDumper.represent_list)

    # Structure YAML data to match fetch_spheres.yaml
    yaml_data = {}
    for link_name, spheres in sphere_dict.items():
        yaml_data[link_name] = spheres

    with open(file_path, 'w') as f:
        # Write comment and data to match fetch_spheres.yaml
        for link_name in sphere_dict.keys():
            if link_name == "base_link":
                f.write("# The base of the robot\n")
            else:
                f.write(f"# the {link_name} of the robot\n")
            yaml.dump({link_name: sphere_dict[link_name]}, f, Dumper=CustomDumper, sort_keys=False, default_flow_style=None, allow_unicode=True)
            f.write("\n")  # Add newline between sections if multiple links
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

# Sphere data for Husky
sphere_data = {}

# Add spheres for base_link (chassis) with original grid and spacing
base_grid = generate_grid(
    center=[0.0, 0.0, 0.2],  # Original center
    x_count=5, y_count=4, z_count=2,  # Original grid counts
    spacing=[0.2, 0.2, 0.2]  # Original spacing
)
sphere_data["base_link"] = add_spheres_to_link("base_link", base_grid, radius=0.15)  # Original radius

# Export YAML
save_yaml(sphere_data, "husky_spheres.yaml")

# Visualize
visualize_husky_and_spheres(sphere_data, urdf_path="husky.urdf")