import os

def create_output_directories(OS_PATH, DATA_SET, sensor_ids):
    # Create needed dataset subfolder for output_path
    output_path = os.path.join(f"{OS_PATH}/output", DATA_SET) + "/"
    os.makedirs(output_path, exist_ok=True)

    # Create 'stats' subfolder in output_path
    stats_path = os.path.join(output_path, "stats")
    os.makedirs(stats_path, exist_ok=True)

    # Create subfolders for sensors in output_path
    for sensor_id_str in sensor_ids:
        sensor_id_str = str(int(sensor_id_str)) if isinstance(sensor_id_str, float) and sensor_id_str.is_integer() else str(sensor_id_str)
        sensor_folder = os.path.join(output_path, "sensors", f"sensor_{sensor_id_str}")
        if not os.path.exists(sensor_folder):
            os.makedirs(sensor_folder)

    return output_path