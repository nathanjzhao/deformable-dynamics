#!/bin/bash

# Configuration
num_objects=5  # Number of objects (columns)
output_file="grid.html"
data_dir="data"  # Base directory for generated data
config_file="sim/generate/config/template.yaml"
max_retries=3  # Maximum number of retries for each cell

# Function to generate a random scene ID
generate_scene_id() {
    echo "scene_$(date +%s%N | md5sum | head -c 10)"
}

# Function to update scene_id in the YAML file
update_scene_id() {
    local scene_id=$1
    sed -i "s/^scene_id:.*/scene_id: $scene_id/" "$config_file"
}

# Function to run simulation and return the path of the generated GIFs
run_simulation() {
    local scene_id=$1
    
    update_scene_id "$scene_id"
    # Redirect stdout and stderr to /dev/null
    python sim/generate.py > /dev/null 2>&1
    
    # Only return paths to the GIF files
    find "$data_dir/$scene_id" -name "visualization_grasp_*.gif" | sort -n
}

# HTML header
cat << EOF > "$output_file"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Actions Grid</title>
    <style>
        table { border-collapse: collapse; width: 100%; }
        td { padding: 10px; border: 1px solid #ddd; text-align: center; vertical-align: top; }
        img { max-width: 200px; max-height: 200px; margin: 5px; }
        .scene-id { font-family: monospace; margin-top: 10px; }
    </style>
</head>
<body>
    <table>
        <tr>
EOF

# Generate table columns (one per object)
for ((j=1; j<=num_objects; j++)); do
    scene_id=$(generate_scene_id)
    gif_paths=()
    
    for ((attempt=1; attempt<=max_retries; attempt++)); do
        # Read the output of run_simulation into an array
        mapfile -t gif_paths < <(run_simulation "$scene_id")
        
        if [ ${#gif_paths[@]} -gt 0 ]; then
            break
        else
            echo "Attempt $attempt failed for $scene_id. Retrying..." >&2
        fi
    done
    
    if [ ${#gif_paths[@]} -gt 0 ]; then
        echo "            <td>" >> "$output_file"
        for gif_path in "${gif_paths[@]}"; do
            echo "                <img src=\"$gif_path\" alt=\"$scene_id action\">" >> "$output_file"
        done
        echo "                <div class=\"scene-id\">$scene_id</div>" >> "$output_file"
        echo "            </td>" >> "$output_file"
    else
        echo "            <td>Failed to generate GIFs for $scene_id after $max_retries attempts</td>" >> "$output_file"
    fi
done

# HTML footer
cat << EOF >> "$output_file"
        </tr>
    </table>
</body>
</html>
EOF

echo "HTML grid generated in $output_file" >&2
