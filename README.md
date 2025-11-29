# generative-modeling
- docs: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition

## 환경 설정 (macOS Apple Silicon)

### 1. Python 3.11 설치
```bash
brew install python@3.11
```

### 2. 가상환경 생성 및 활성화
```bash
/opt/homebrew/bin/python3.11 -m venv myenv
source myenv/bin/activate
```

### 3. TensorFlow 설치
```bash
pip install tensorflow-macos
pip install tensorflow-metal  # GPU 가속용
```

### 4. 설치 확인
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 사용법

- 프로젝트 시작 시
    ```bash
    cd ~/Desktop/programs/study/ai/generative-modeling
    source myenv/bin/activate
    ```

- 종료 시
    ```bash
    deactivate
    ```
