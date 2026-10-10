# One image for local runs and for Lambda. Build arm64 for Graviton:
#   docker buildx build --platform linux/arm64 -f vision/Dockerfile .
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# YOLOX-S from the OpenCV Model Zoo (Apache 2.0). Fetched at build time and
# checked by hash, so a clean clone rebuilds the same image.
COPY vision/detector.py /tmp/detector_meta.py
RUN python - <<'EOF'
import hashlib, importlib.util, os, urllib.request
spec = importlib.util.spec_from_file_location("m", "/tmp/detector_meta.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dst = os.path.join(os.environ["LAMBDA_TASK_ROOT"], "models", m.MODEL_FILE)
os.makedirs(os.path.dirname(dst), exist_ok=True)
urllib.request.urlretrieve(m.MODEL_URL, dst)
digest = hashlib.sha256(open(dst, "rb").read()).hexdigest()
assert digest == m.MODEL_SHA256, f"model hash mismatch: {digest}"
print("model ok", os.path.getsize(dst))
EOF

COPY vision ${LAMBDA_TASK_ROOT}/vision

# The Lambda handler CMD is added with the AWS stack. docker-compose.yml
# overrides the entrypoint to run the CLIs locally.
