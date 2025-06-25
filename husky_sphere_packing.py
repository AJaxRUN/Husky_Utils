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
    # Custom YAML Dumper to format lists with proper spacing and precision
    class CustomDumper(yaml.Dumper):
        def represent_list(self, data):
            if isinstance(data, list) and all(isinstance(item, (int, float)) for item in data):
                formatted = [f"{item: .4f}" if isinstance(item, float) else str(item) for item in data]
                return self.represent_sequence('tag:yaml.org,2002:seq', formatted, flow_style=True)
            return super().represent_list(data)

    CustomDumper.add_representer(list, CustomDumper.represent_list)

    # Add comments to the YAML structure
    yaml_data = {}
    for link_name, spheres in sphere_dict.items():
        if link_name == "base_link":
            comment = "The base of the robot"
        elif link_name == "torso_lift_link":
            comment = "the torso of the robot"
        elif link_name == "head_pan_link":
            comment = "the head of the robot"
        else:
            comment = f"the {link_name} of the robot"
        yaml_data[f"# {comment}\n{link_name}"] = spheres

    with open(file_path, 'w') as f:
        yaml.dump(yaml_data, f, Dumper=CustomDumper, sort_keys=False, default_flow_style=None, allow_unicode=True)
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


sphere_data = {}

# Add grid
chassis_grid = generate_grid(center=[0.0, 0.0, 0.2], x_count=5, y_count=4, z_count=2, spacing=[0.2, 0.2, 0.2])
sphere_data["chassis_link"] = add_spheres_to_link("chassis_link", chassis_grid, radius=0.15)

# Export YAML
save_yaml(sphere_data, "husky_spheres.yaml")

# Visualize
visualize_husky_and_spheres(sphere_data, urdf_path="husky.urdf")