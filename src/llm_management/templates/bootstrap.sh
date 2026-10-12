#!/bin/bash
# Runs as root on the disposable Ubuntu 24.04 VM, via cloud-init.
set -euo pipefail
trap 'touch /opt/llm-template/failed' ERR
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y docker.io "$NVIDIA_DRIVER_PACKAGE" curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
# Avoid the documented GPU access loss on systemd daemon-reload.
python3 - <<'PY'
import json
from pathlib import Path
path = Path('/etc/docker/daemon.json')
config = json.loads(path.read_text())
config['exec-opts'] = ['native.cgroupdriver=cgroupfs']
path.write_text(json.dumps(config))
PY
systemctl restart docker
modprobe nvidia
modprobe nvidia_uvm
nvidia-smi
mkdir -p /opt/systemone /opt/systemone/model-cache
docker pull "$SYSTEMONE_IMAGE"
docker image inspect --format '{{index .RepoDigests 0}}' "$SYSTEMONE_IMAGE" > /opt/systemone/image
# Persist only non-secret runtime settings, with shell-safe quoting.
for setting in SYSTEMONE_BACKEND SYSTEMONE_MODEL SYSTEMONE_REVISION SYSTEMONE_MAX_LENGTH SYSTEMONE_PORT; do
  printf '%s=%q\n' "$setting" "${!setting}"
done > /opt/systemone/runtime.env
cat > /opt/systemone/start.sh <<'SCRIPT'
#!/bin/bash
set -euo pipefail
source /opt/systemone/runtime.env
image=$(cat /opt/systemone/image)
exec docker run --rm --name systemone --pull never --gpus all --shm-size=2g \
  -p "127.0.0.1:$SYSTEMONE_PORT:8000" \
  -e SYSTEMONE_BACKEND="$SYSTEMONE_BACKEND" \
  -e SYSTEMONE_MODEL="$SYSTEMONE_MODEL" \
  -e SYSTEMONE_REVISION="$SYSTEMONE_REVISION" \
  -e SYSTEMONE_MAX_LENGTH="$SYSTEMONE_MAX_LENGTH" \
  -e HF_HUB_OFFLINE=1 \
  -v /opt/systemone/model-cache:/data/huggingface "$image"
SCRIPT
chmod 755 /opt/systemone/start.sh
cat > /etc/systemd/system/systemone.service <<'UNIT'
[Unit]
Description=System One Clef server
After=docker.service
Requires=docker.service
[Service]
ExecStart=/opt/systemone/start.sh
ExecStop=/usr/bin/docker stop -t 30 systemone
Restart=on-failure
RestartSec=10
TimeoutStopSec=45
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable systemone.service
# Exercise CUDA itself, rather than only the driver's nvidia-smi interface.
docker run --rm --gpus all "$SYSTEMONE_IMAGE" python -c \
  'import torch; print(torch.__version__); print(torch.zeros(1, device="cuda")); print(torch.cuda.get_device_name())'
# The first verification downloads the weights; the boot service will be offline.
docker run -d --name systemone --gpus all --shm-size=2g \
  -p 127.0.0.1:$SYSTEMONE_PORT:8000 \
  -e SYSTEMONE_BACKEND="$SYSTEMONE_BACKEND" \
  -e SYSTEMONE_MODEL="$SYSTEMONE_MODEL" \
  -e SYSTEMONE_REVISION="$SYSTEMONE_REVISION" \
  -e SYSTEMONE_MAX_LENGTH="$SYSTEMONE_MAX_LENGTH" \
  -v /opt/systemone/model-cache:/data/huggingface "$SYSTEMONE_IMAGE"
