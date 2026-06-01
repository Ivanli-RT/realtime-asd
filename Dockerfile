ARG BASE_IMAGE=10.51.33.201:30002/navi_project/ros:noetic-l4t-r36.3.0
FROM ${BASE_IMAGE}

ARG ROS_DISTRO=noetic
ARG ROS_ROOT=/opt/ros/${ROS_DISTRO}
ARG ROS_SETUP_PATH=/ros_noetic/catkin_ws/devel/setup.bash
ARG ROS_FALLBACK_SETUP_PATH=${ROS_ROOT}/setup.bash

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    ROS_DISTRO=${ROS_DISTRO} \
    ROS_ROOT=${ROS_ROOT} \
    ROS_SETUP_PATH=${ROS_SETUP_PATH} \
    ROS_FALLBACK_SETUP_PATH=${ROS_FALLBACK_SETUP_PATH}

ARG INSTALL_APT=0
ARG INSTALL_PY_DEPS=1
ARG INSTALL_NVIDIA_APT_SOURCES=0
ARG NVIDIA_JETSON_APT_DIST=r36.4
ARG NVIDIA_JETSON_SOC=t234
ARG INSTALL_CUDNN=0
ARG CUDNN_APT_PACKAGES=cudnn9-cuda-12
ARG REQUIRE_ROS=1
ARG ULTRALYTICS_VERSION=8.4.43
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG UPGRADE_PIP=0
ARG PIP_RETRIES=2
ARG PIP_TIMEOUT=30
ARG INSTALL_PYCUDA=1
ARG PYCUDA_VERSION=2026.1

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
RUN if [ "$INSTALL_APT" = "1" ]; then \
            set -eux; \
            mkdir -p /tmp/apt-sources-backup; \
            find /etc/apt/sources.list.d -maxdepth 1 -type f \( -name 'ros*.list' -o -name '*ros*ubuntu*.list' \) -exec mv {} /tmp/apt-sources-backup/ \; 2>/dev/null || true; \
            apt-get update && apt-get install -y --no-install-recommends \
                ca-certificates \
                curl \
                gnupg \
            ; \
            if [ "$INSTALL_NVIDIA_APT_SOURCES" = "1" ]; then \
                mkdir -p /etc/apt/keyrings; \
                curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
                    | gpg --batch --yes --dearmor -o /etc/apt/keyrings/nvidia-jetson.gpg; \
                echo "deb [signed-by=/etc/apt/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/common ${NVIDIA_JETSON_APT_DIST} main" > /etc/apt/sources.list.d/nvidia-jetson.list; \
                echo "deb [signed-by=/etc/apt/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/${NVIDIA_JETSON_SOC} ${NVIDIA_JETSON_APT_DIST} main" >> /etc/apt/sources.list.d/nvidia-jetson.list; \
                apt-get update; \
            else \
                echo "[Dockerfile] Skip NVIDIA Jetson apt sources (INSTALL_NVIDIA_APT_SOURCES=0)"; \
            fi; \
            apt-get install -y --no-install-recommends \
                python3-pip \
                python3-dev \
                python3-opencv \
                python3-setuptools \
                python3-wheel \
                build-essential \
                git \
                ffmpeg \
                libglib2.0-0 \
                libgl1 \
                libopenblas0 \
            ; \
            if [ "$INSTALL_CUDNN" = "1" ]; then \
                apt-get install -y --no-install-recommends $CUDNN_APT_PACKAGES; \
            else \
                echo "[Dockerfile] Skip cuDNN install (INSTALL_CUDNN=0)"; \
            fi; \
            rm -rf /var/lib/apt/lists/*; \
            find /tmp/apt-sources-backup -maxdepth 1 -type f -exec mv {} /etc/apt/sources.list.d/ \; 2>/dev/null || true; \
        else \
            echo "[Dockerfile] Skip apt install (INSTALL_APT=0)"; \
        fi

RUN if [ -f "$ROS_SETUP_PATH" ]; then \
        echo "[Dockerfile] ROS environment found: $ROS_SETUP_PATH"; \
    elif [ -f "$ROS_FALLBACK_SETUP_PATH" ]; then \
        echo "[Dockerfile] ROS environment found at fallback: $ROS_FALLBACK_SETUP_PATH"; \
    elif command -v rostopic >/dev/null 2>&1 && python3 -c "import rospy" >/dev/null 2>&1; then \
        echo "[Dockerfile] ROS1 tools found in PATH, but setup file is missing: $ROS_SETUP_PATH"; \
    elif [ "$REQUIRE_ROS" = "1" ]; then \
        echo "[Dockerfile] ERROR: ROS environment not found: $ROS_SETUP_PATH"; \
        echo "[Dockerfile] Use a ROS1 Noetic base image, or set ROS_DISTRO/ROS_ROOT/ROS_SETUP_PATH to the ROS1 install provided by the base image."; \
        echo "[Dockerfile] Existing candidates:"; \
        find /opt /usr/local /workspace /ros_noetic -maxdepth 6 -type f \( -name setup.bash -o -name local_setup.bash \) 2>/dev/null | sort || true; \
        echo "[Dockerfile] rostopic path:"; command -v rostopic || true; \
        exit 1; \
    else \
        echo "[Dockerfile] WARNING: ROS environment not found: $ROS_SETUP_PATH"; \
        echo "[Dockerfile] Existing candidates:"; \
        find /opt /usr/local /workspace /ros_noetic -maxdepth 6 -type f \( -name setup.bash -o -name local_setup.bash \) 2>/dev/null | sort || true; \
        echo "[Dockerfile] rostopic path:"; command -v rostopic || true; \
        echo "[Dockerfile] Continue build. Set ASD_REQUIRE_ROS=1 after confirming the base image has ROS1."; \
    fi

WORKDIR /workspace/asd_runtime_slim

COPY packages/ /tmp/local-packages/

# Deploy local zj_humanoid ROS1 type/service packages from the bundled Makeself .run archive.
# The base image provides ROS from a prebuilt workspace, not from ROS apt packages, so
# installing the embedded focal/noetic debs with apt would leave unresolved dependencies.
RUN set -eux; \
    TYPES_RUN="$(ls -1 /tmp/local-packages/system/zj_humanoid_types*.run 2>/dev/null | sort | tail -n 1 || true)"; \
    if [ -n "$TYPES_RUN" ]; then \
        chmod +x "$TYPES_RUN"; \
        rm -rf /tmp/zj_humanoid_types; \
        mkdir -p /tmp/zj_humanoid_types; \
        sh "$TYPES_RUN" --noexec --target /tmp/zj_humanoid_types --noprogress --nochown; \
        for deb in /tmp/zj_humanoid_types/*.deb; do \
            dpkg-deb -x "$deb" /; \
        done; \
        rm -rf /tmp/zj_humanoid_types; \
    else \
        echo "[Dockerfile] no zj_humanoid types .run found under /tmp/local-packages/system"; \
    fi

# Install Python dependencies at build time (one-time).
COPY requirements.common.txt /tmp/requirements.common.txt
RUN if [ "$INSTALL_PY_DEPS" = "1" ]; then \
    python3 -m pip config unset global.index-url || true && \
    python3 -m pip config unset global.extra-index-url || true && \
    if [ "$UPGRADE_PIP" = "1" ]; then \
        python3 -m pip install --no-cache-dir --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" "pip<24.1"; \
    else \
        echo "[Dockerfile] skip pip upgrade (UPGRADE_PIP=0)"; \
    fi && \
    for pattern in 'torch-*.whl' 'torchaudio-*.whl' 'torchvision-*.whl'; do \
        WHEEL="$(ls -1 /tmp/local-packages/python/${pattern} 2>/dev/null | sort | tail -n 1 || true)"; \
        if [ -n "$WHEEL" ]; then \
            python3 -m pip install --no-cache-dir --no-deps --no-index --find-links /tmp/local-packages/python "$WHEEL"; \
        fi; \
    done && \
    python3 -m pip install --no-cache-dir --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" -r /tmp/requirements.common.txt && \
    python3 -m pip install --no-cache-dir --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" --no-deps "ultralytics==${ULTRALYTICS_VERSION}" && \
    if [ "$INSTALL_PYCUDA" = "1" ]; then \
        if python3 -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('pycuda') else 1)"; then \
            echo "[Dockerfile] pycuda already installed"; \
        else \
            python3 -m pip install --no-cache-dir --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" "pycuda==${PYCUDA_VERSION}"; \
        fi; \
        python3 -c "import pycuda, pycuda.driver as cuda; print('[build] pycuda=', pycuda.VERSION_TEXT); print('[build] pycuda_driver_import=ok')"; \
        python3 -c "import importlib.util; ok = importlib.util.find_spec('tensorrt') is not None; print('[build] tensorrt_present=', ok); assert ok"; \
    else \
        echo "[Dockerfile] Skip pycuda install (INSTALL_PYCUDA=0)"; \
    fi && \
    python3 -c "import importlib.util; missing=[m for m in ('torch','torchvision') if importlib.util.find_spec(m) is None]; print('[build] required_torch_modules_missing=', missing); assert not missing" && \
    if [ "$INSTALL_CUDNN" = "1" ]; then \
        python3 -c "import torch; print('[build] torch=', torch.__version__)"; \
    else \
        echo "[Dockerfile] skip torch import check at build time (INSTALL_CUDNN=0; expect host cuDNN mount at runtime)"; \
    fi && \
    python3 -c "import importlib.util; ok = importlib.util.find_spec('ultralytics') is not None; print('[build] ultralytics_present=', ok); assert ok"; \
    else \
        echo "[Dockerfile] Skip python deps install (INSTALL_PY_DEPS=0)"; \
    fi

# Install local multimodal_process_tap wheel when provided.
RUN PROCESS_TAP_WHL="$(ls -1 /tmp/local-packages/python/multimodal_process_tap-*.whl 2>/dev/null | sort | tail -n 1 || true)" && \
    if [ -n "$PROCESS_TAP_WHL" ]; then \
        python3 -m pip install --no-cache-dir --no-deps "$PROCESS_TAP_WHL" && \
        python3 -c "import importlib.util, inspect; ok = importlib.util.find_spec('process_tap') is not None; print('[build] process_tap_present=', ok); assert ok; from process_tap.integrations import maybe_create_asd_stream_capture; src = inspect.getsource(maybe_create_asd_stream_capture); assert 'ASD_STREAM_CAPTURE_WRITE_PREVIEW_VIDEO' in src, 'stale process_tap wheel installed'; print('[build] process_tap_capture_preview_switch=present')"; \
    else \
        echo "[Dockerfile] no process_tap wheel found under /tmp/local-packages/python, skip process_tap install"; \
    fi

# Install ASD SDK from prebuilt wheel if present (recommended for decoupled SDK usage).
RUN ASD_SDK_WHL="$(ls -1 /tmp/local-packages/python/asd_sdk-*.whl 2>/dev/null | sort | tail -n 1 || true)" && \
    if [ -n "$ASD_SDK_WHL" ]; then \
        python3 -m pip install --no-cache-dir "$ASD_SDK_WHL" && \
        python3 -c "import asd_sdk; print('[build] asd_sdk installed:', hasattr(asd_sdk, 'ASDSDK'))"; \
    else \
        echo "[Dockerfile] no sdk wheel found under /tmp/local-packages/python, skip sdk install"; \
    fi

CMD ["bash"]
