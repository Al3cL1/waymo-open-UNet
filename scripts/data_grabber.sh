#!/bin/bash

# Display usage
function usage {
    printf "%s\n" "Usage:"
    printf "%s\n" "$0 data_set destination_folder [-f file_list]"
    printf "%s\n" "Options:"
    printf "%s\n" "-f file_list      Provide a text file with list of files to download"
    exit 1
}

# Check number of arguments is valid
if [ $# -lt 2 ]; then
    usage
fi

# Parse command-line arguments
data_set="$1"
destination_folder="$2"
shift 2

file_list=""

# Process optional file arguments
while getopts "f:" opt; do
    case $opt in
        f)
            file_list="$OPTARG"
            ;;
        *)
            usage
            ;;
    esac
done

base_path="gs://waymo_open_dataset_v_2_0_1/$data_set"
gsutil_cmd="gsutil -m cp"

# Function to create directory if it doesn't exist
create_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi
}

# Check if a file list is provided
if [ -n "$file_list" ]; then
    printf "%s\n" "Downloading files listed in $file_list to $destination_folder"
    while IFS= read -r file; do
        file=$(echo "$file" | tr -d '\r')
        if [ -n "$file" ]; then
            data_types=$(gsutil ls "$base_path/" | grep '/$' | tail -n +2)
            for data_type in $data_types; do
                data_type_name=$(basename "$data_type")
                full_path="${data_type}${file}"
                data_type_folder="${destination_folder}/${data_set}/${data_type_name}"
                create_directory "$data_type_folder"
                $gsutil_cmd "$full_path" "$data_type_folder/"
            done
        fi
    done < "$file_list"
else
    # Download the entire data_set
    full_path="$base_path/*"
    printf "%s\n" "Downloading contents of $base_path to $destination_folder..."
    $gsutil_cmd -r "$full_path" "$destination_folder"
fi

if [ $? -eq 0 ]; then
    printf "%s\n" "Download completed successfully"
else
    printf "%s\n" "Failed to download files" >&2
    exit 1
fi
