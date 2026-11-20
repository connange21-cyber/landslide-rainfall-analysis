# 원자료 전처리

강우·지형·토지피복을 행정구역·강우 이벤트에 연결한다. 가공 입력의 학습·TEST·SHAP는 [모델 재현 패키지](../experiments/model_comparison/README.md)를 사용한다.

## 입력

산림청·한국도로공사의 산사태 기록, 기상청 AWS·ASOS·융합강우, 환경공간정보서비스 토지피복, 별도 제공된 5m DEM·행정구역 경계를 사용한다.

루트 `base/landslide`, `base/stations`, `base/boundaries`, `base/landcover`에 목록·관측소 이력·경계·피복을 준비한다. `config/`의 외부 경로 `/data/RAIN_DATAS/지상관측`, `/data/DEM_5m`을 실제 위치에 맞춘다. API 인증정보는 환경변수나 숨김 입력으로 제공한다.

## 처리 흐름

| `scripts/` 하위 폴더 | 역할 |
| --- | --- |
| `station_linking/` | 발생 기록을 행정구역·관측소에 연결 |
| `hourly_rainfall/` | 위치 이력을 반영한 시간강우 구축 |
| `rainfall_events/` | 강우 이벤트와 발생 기간 연결 |
| `cumulative_rainfall/` | 종료일 기준 누적강우 계산 |
| `terrain/` | DEM 경사와 행정구역 통계 산출 |
| `temporal_controls/` | 같은 지역의 다른 강우 이벤트 선택 |
| `spatial_controls/` | 같은 시간대의 다른 지역 선택 |
| `land_cover/` | 피복 도엽 확보·연도 연결·면적 집계 |
| `data_integration/` | 강우·경사·피복과 표본 통합 |
| `fusion_rainfall/` | 융합기상 격자 수집·단위 해석 |
| `rainfall_summary/` | 구역별 융합강우 요약 |
| `event_matching_visualization/` | 대표 이벤트 재연결·시각화 |
| `sample_construction/` | 발생·대조 표본 재구축·추가 관측소 후보 연결 |

발생 기록 A에 같은 구역의 다른 이벤트 B, 같은 시간대의 다른 구역 C를 연결한다. B·C는 발생 기록 미포착 표본이다. 발생 시각이 없으면 최대 시간융합강우로 대표 이벤트를 추정하고 불명확한 사례는 미확정으로 남긴다.

## 실행과 저장

원자료·의존성을 준비한 뒤 루트에서 실행한다. 일부 자료를 이용한 시작 예시이며 후속 코드는 앞선 산출물이 필요하다.

```bash
python -m pip install -r requirements.txt
python scripts/station_linking/01_link_landslides_stations.py --limit 2
python scripts/hourly_rainfall/02_build_rainfall.py --smoke
```

`output/01/...`, `log/01/...` 등의 숫자는 이전 단계 입출력을 연결하는 내부 호환 식별자다. 코드 폴더의 실행 순서를 뜻하는 공개 목차가 아니다. 분석 결과는 `output/`, 실행·API·QC·오류 기록은 `log/`에 분리한다. 원자료와 앞선 산출물은 읽기 전용이며 기존 결과 교체에는 코드가 요구하는 명시적 옵션을 사용한다.

원자료·원 DEM·API 응답은 미배포하여 저장소만으로 전처리 전체를 재실행할 수 없다. REVIEW는 보존한다. 발생 후 강우가 포함될 수 있어 실시간 사전 예측 검증으로 해석하지 않는다.
