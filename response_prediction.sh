#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config.yaml> <param1=value1> [param2=value2] ..."
    exit 1
fi

# Store the config file path
config_file="$1"

# Create a temporary config file
temp_config=$(mktemp)

# Copy the original config file to the temporary file
cp "$config_file" "$temp_config"

# Shift the arguments to start from the second argument
shift

# Create a Python script to update the YAML file
update_script=$(mktemp)

cat << EOF > "$update_script"
import yaml

with open("$temp_config", "r") as f:
    config = yaml.safe_load(f)

for param in "$@".split():
    keys = param.split("=")[0].split(".")
    value = param.split("=")[1]
    current_level = config
    for key in keys[:-1]:
        current_level = current_level.setdefault(key, {})
    current_level[keys[-1]] = value

with open("$temp_config", "w") as f:
    yaml.dump(config, f)
EOF

# Run the Python script to update the temporary config file
python "$update_script" "$@"

# Remove the temporary Python script
rm "$update_script"

# Run your Python experiment with the updated temporary config file
python main.py --config "$temp_config"

# Clean up the temporary config file
rm "$temp_config"