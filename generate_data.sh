#!/bin/bash

# Configuration
dataset_name="smaller_voxels"  # Configurable dataset folder name
number_of_objects=100    # Number of scenes to generate
config_file="sim/generate/config/template.yaml"
max_retries=3           # Maximum number of retries for each scene
data_dir="data"         # Base directory for generated data

# Function to generate a random scene ID with dataset prefix
generate_scene_id() {
    local random_suffix=$(date +%s%N | md5sum | head -c 10)
    echo "${dataset_name}/scene_${random_suffix}"
}

# Function to update scene_id in the YAML file
update_scene_id() {
    local scene_id=$1
    sed -i "s|^scene_id:.*|scene_id: $scene_id|" "$config_file"
}

# Function to run simulation
run_simulation() {
    local scene_id=$1
    update_scene_id "$scene_id"
    
    # Create the dataset directory if it doesn't exist
    mkdir -p "$data_dir/$dataset_name"
    
    # Run the simulation
    python sim/generate.py > /dev/null 2>&1
    
    # Check if the simulation generated any files
    if [ -d "$data_dir/$scene_id" ]; then
        return 0
    else
        return 1
    fi
}

# Main loop to generate scenes
for ((i=1; i<=number_of_objects; i++)); do
    scene_id=$(generate_scene_id)
    success=false
    
    echo "Generating scene $i of $number_of_objects: $scene_id"
    
    for ((attempt=1; attempt<=max_retries; attempt++)); do
        if run_simulation "$scene_id"; then
            echo "Successfully generated scene $scene_id after $attempt attempts"
            success=true
            break
        else
            echo "Attempt $attempt failed for $scene_id. Retrying..."
        fi
    done
    
    if ! $success; then
        echo "Failed to generate scene $scene_id after $max_retries attempts"
    fi
done

echo "Dataset generation complete in $data_dir/$dataset_name"