import yaml
import pybullet as p
import pybullet_data
import time

def load_yaml(file_path='husky_spheres.yaml'):
    """Load sphere data from YAML file."""
    with open(file_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML: {e}")
            return {}

def visualize_urdf_and_spheres(urdf_path='husky.urdf', yaml_path='husky_spheres.yaml'):
    """Visualize the URDF robot model and spheres from YAML in PyBullet."""
    # Connect to PyBullet GUI
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load the ground plane and robot URDF
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(urdf_path, [0, 0, 0.1], useFixedBase=False)
    
    # Load sphere data from YAML
    sphere_data = load_yaml(yaml_path)
    if not sphere_data:
        print("No sphere data loaded. Exiting visualization.")
        p.disconnect()
        return
    
    # Visualize spheres for each link
    for link_name, spheres in sphere_data.items():
        if not isinstance(spheres, list):
            print(f"Skipping invalid sphere data for {link_name}")
            continue
        for sphere in spheres:
            centers = sphere.get("center", [])
            radius = sphere.get("radius", 0.1)
            if not centers or not isinstance(centers, list):
                print(f"Skipping invalid center data for {link_name}")
                continue
            if not isinstance(centers[0], list):
                centers = [centers]  # Ensure single centers are treated as list of lists
            for center in centers:
                # Ensure center coordinates are floats
                try:
                    center = [float(x) for x in center]
                except (ValueError, TypeError):
                    print(f"Invalid center coordinates for {link_name}: {center}")
                    continue
                # Create semi-transparent red sphere
                vis_id = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    rgbaColor=[1, 0, 0, 0.6]  # Red, semi-transparent
                )
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,  # No collision
                    baseVisualShapeIndex=vis_id,
                    basePosition=center
                )
    
    print("Visualization loaded. Press Ctrl+C to exit.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(0.01)  # Control rendering speed
    except KeyboardInterrupt:
        print("Exiting visualization.")
        p.disconnect()

# Example usage
if __name__ == "__main__":
    visualize_urdf_and_spheres(urdf_path="husky.urdf", yaml_path="husky_spheres.yaml")