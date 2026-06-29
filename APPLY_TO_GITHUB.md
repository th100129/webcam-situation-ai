# 기존 GitHub 프로젝트에 안전하게 반영하는 순서

이 폴더는 이전에 만들었던 `webcam-situation-ai_modularized`와 **섞어 쓰면 안 됩니다.**

## 1. 새 폴더에서 먼저 단독 실행

```powershell
python -m pip install -r requirements.txt
python download_models.py
python main.py
```

얼굴 메쉬, 손 랜드마크, 자세 선이 보이는지 확인합니다.

## 2. 기존 저장소를 백업 커밋

기존 `webcam-situation-ai` 폴더 터미널에서:

```powershell
git switch dev
git add .
git commit -m "backup: before full modular refactor"
```

## 3. 새 모듈화 폴더의 파일을 기존 저장소 최상위에 복사

기존에 있던 루트 파일과 새 구조의 중복 파일은 새 버전으로 교체합니다.

특히 아래 루트 파일은 새 위치로 이동하므로, 예전 파일을 남겨두지 않습니다.

```text
face_mesh_detector.py      -> detectors/face_mesh_detector.py
hand_detector.py           -> detectors/hand_detector.py
pose_detector.py           -> detectors/pose_detector.py
focus_analyzer.py          -> focus/focus_analyzer.py
focus_action_manager.py    -> focus/focus_action_manager.py
mission_manager.py         -> mission/mission_manager.py
llm_manager.py             -> feedback/llm_manager.py
prompt_templates.py        -> feedback/prompt_templates.py
metrics_tracker.py         -> utils/metrics_tracker.py
report_generator.py        -> utils/report_generator.py
session_logger.py          -> utils/session_logger.py
```

## 4. Git 저장

```powershell
git add .
git status
git commit -m "refactor: modularize full focus mission system"
git push origin dev
```
