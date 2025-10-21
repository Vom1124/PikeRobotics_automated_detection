import yaml

METADATA_FILE = "metadata.yaml"

def fix_times(obj):
    """Convert all duration/starting_time to Humble-compatible nanoseconds format."""
    if isinstance(obj, dict):
        # Convert top-level duration
        if "duration" in obj and isinstance(obj["duration"], int):
            obj["duration"] = {"nanoseconds": obj["duration"]}
        # Convert top-level starting_time
        if "starting_time" in obj and isinstance(obj["starting_time"], int):
            obj["starting_time"] = {"nanoseconds_since_epoch": obj["starting_time"]}
        # Recurse
        for k, v in obj.items():
            fix_times(v)
    elif isinstance(obj, list):
        for item in obj:
            fix_times(item)

with open(METADATA_FILE, "r") as f:
    data = yaml.safe_load(f)

info = data.get("rosbag2_bagfile_information", {})

# 1) Ensure custom_data is at least empty dict
if info.get("custom_data", None) is None:
    info["custom_data"] = {}

# 2) Force ros_distro to string
if "ros_distro" in info:
    info["ros_distro"] = str(info["ros_distro"])

# 3) Apply nanoseconds format to all durations and starting_times
fix_times(info)

# 4) Force QoS reliability
for t in info.get("topics_with_message_count", []):
    meta = t.get("topic_metadata", {})
    if "offered_qos_profiles" in meta:
        # Keep it YAML string like Humble
        meta["offered_qos_profiles"] = "- reliability: 1\n  history: 3\n  depth: 0\n  durability: 2\n  deadline:\n    sec: 9223372036\n    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n    nsec: 854775807\n  liveliness: 1\n  liveliness_lease_duration:\n    sec: 9223372036\n    nsec: 854775807\n  avoid_ros_namespace_conventions: false"

# Write back
with open(METADATA_FILE, "w") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print("✅ Metadata converted to ROS 2 Humble-compatible format.")
