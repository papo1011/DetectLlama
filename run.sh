EXE_NAME="DetectLlama"
BUILD_DIR="build"
MODEL_DIR="models"
INPUT_DIR="inputs"

ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo "'$ENV_FILE' not found. Please create it"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

EXECUTABLE="./$BUILD_DIR/$EXE_NAME"
MODEL_PATH="./$MODEL_DIR/$MODEL_NAME"
INPUT_PATH="./$INPUT_DIR/$INPUT_FILE"

MODEL_BASE_URL="https://$HF_TOKEN@huggingface.co"
MODEL_URL="$MODEL_BASE_URL/$MODEL_HF_PATH?download=true"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found on disk. Download..."
    mkdir -p "$MODEL_DIR"
    wget --show-progress -c "$MODEL_URL" -O "$MODEL_PATH"
fi

$EXECUTABLE -m "$MODEL_PATH" -f "$INPUT_PATH"
