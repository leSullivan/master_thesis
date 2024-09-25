REMOTE_SERVER="sc-uni"
REMOTE_DIR="/work/jw018njay-model_checkpoints/"
LOCAL_DIR="./tb_logs"

mkdir -p "$LOCAL_DIR"


ssh $REMOTE_SERVER "find $REMOTE_DIR -name 'events.out.tfevents.*' -o -name 'hparams.yaml'" | while IFS= read -r remote_file; do
    relative_path="${remote_file#$REMOTE_DIR}"

    local_file="$LOCAL_DIR/$relative_path"
    local_dir=$(dirname "$local_file")
    mkdir -p "$local_dir"

    scp "$REMOTE_SERVER:$remote_file" "$local_file" </dev/null
    if [ $? -eq 0 ]; then
        echo "File found. Copying..."
    else
        echo "Failed to copy: $remote_file"
    fi
done

echo "Copy process completed!"
