REMOTE_SERVER=""
REMOTE_DIR=""
LOCAL_DIR=""

mkdir -p "$LOCAL_DIR"

ssh $REMOTE_SERVER "find $REMOTE_DIR -name 'events.out.tfevents.*'" | while IFS= read -r remote_tfevent_file; do
    # Derive the relative path for the tfevents file
    relative_path="${remote_tfevent_file#$REMOTE_DIR}"

    # Set up local file and directory paths for the tfevents file
    local_tfevent_file="$LOCAL_DIR/$relative_path"
    local_dir=$(dirname "$local_tfevent_file")
    mkdir -p "$local_dir"

    # Copy the tfevents file
    scp "$REMOTE_SERVER:$remote_tfevent_file" "$local_tfevent_file" </dev/null
    if [ $? -eq 0 ]; then
        echo "Tfevents file copied: $remote_tfevent_file"

        # Construct the corresponding hparams.yaml file path
        remote_hparams_file="$(dirname "$remote_tfevent_file")/hparams.yaml"
        local_hparams_file="$local_dir/hparams.yaml"

        # Copy the hparams.yaml file
        scp "$REMOTE_SERVER:$remote_hparams_file" "$local_hparams_file" </dev/null
        if [ $? -eq 0 ]; then
            echo "Hparams file copied: $remote_hparams_file"
        else
            echo "Failed to copy hparams file: $remote_hparams_file"
        fi
    else
        echo "Failed to copy tfevents file: $remote_tfevent_file"
    fi
done

echo "Copy process completed!"
