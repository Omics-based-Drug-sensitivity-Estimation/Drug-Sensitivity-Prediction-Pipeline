# YoonjinCho

이 프로젝트는 오믹스(omics) 기반의 약물 반응 예측(Drug Sensitivity Estimation)을 위한 코드 및 데이터로 구성되어 있습니다.

## 폴더 구조 및 주요 파일

- `analysis.py`  
  데이터 분석 및 결과 해석을 위한 스크립트입니다.

- `debug_utils.py`  
  디버깅 및 유틸리티 함수가 포함되어 있습니다.

- `models/`  
  다양한 모델 구조와 관련 코드가 포함되어 있습니다.  
  - `model.py`, `model2.py`, `model3.py`: 여러 모델 아키텍처 구현  
  - `PASO/`: 하위 모델 관련 코드

- `data/`  
  실험에 사용되는 데이터와 전처리 코드가 포함되어 있습니다.  
  - `10_fold_data/`: 10-fold 교차검증용 데이터  
  - 다양한 omics 데이터(csv, pkl 등)  
  - `TripleOmics_Drug_Dataset.py`: 데이터셋 생성 및 관리 코드

- `result/`  
  실험 결과 및 생성된 모델 파일이 저장되는 폴더입니다.  
  - `model/`: 학습된 모델 파일 등

- `train/`  
  모델 학습 관련 코드가 포함되어 있습니다.  
  - `train_tqdm.py` 등: 다양한 학습 스크립트

- `utils/`  
  공통적으로 사용하는 함수 및 모듈이 포함되어 있습니다.  
  - `CrossAttention.py`, `DrugEmbedding.py` 등: 네트워크 레이어, 임베딩, loss 등

- `yj/`  
  인코더 등 후속 연구를 위한 추가 실험 코드가 포함되어 있습니다.  
  - `encoder.py`, `encoder2.py`, `encoder3.py` 등