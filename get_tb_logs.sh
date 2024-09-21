REMOTE_SERVER="sc-uni"
REMOTE_DIR="/work/jw018njay-model_checkpoints/"
LOCAL_DIR="./tb_logs"  # Change this to the directory where you want to copy the files

mkdir -p "$LOCAL_DIR"

# Use SSH and find to locate all events.out.tfevents.* files and copy them to the specified local directory
ssh $REMOTE_SERVER "find $REMOTE_DIR -name 'events.out.tfevents.*'" | while IFS= read -r remote_file; do
    echo "Found: $remote_file"
    
    # Get the relative path of the file on the remote server
    relative_path="${remote_file#$REMOTE_DIR}"

    # Recreate the directory structure in the local directory without adjusting the version name
    local_file="$LOCAL_DIR/$relative_path"
    local_dir=$(dirname "$local_file")
    mkdir -p "$local_dir"

    # Use scp to copy the file to the specified local directory and check for success
    scp "$REMOTE_SERVER:$remote_file" "$local_file" </dev/null
    if [ $? -eq 0 ]; then
        echo "Copied: $remote_file to $local_file"
    else
        echo "Failed to copy: $remote_file"
    fi
done

echo "Copy process completed!"
