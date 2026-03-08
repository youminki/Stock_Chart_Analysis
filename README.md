# Rule-Based Stock Backtesting Engine

순수 룰 기반(비-ML) 주식 백테스트 엔진입니다.
데이터 제공자: `yahoo`(기본), `twelve`(Twelve Data API).

## 설치

```bash
cd /Users/psy/Documents/project/stock_test
python3 -m pip install -r requirements.txt
```

## 데이터 제공자 선택

- 기본: `--provider yahoo`
- Twelve Data: `--provider twelve --api-key <YOUR_KEY>`
  - 또는 환경변수: `TWELVE_DATA_API_KEY`

## 명령어

### 1) 단일 티커 백테스트

```bash
python3 main.py --provider yahoo analyze --ticker AAPL
python3 main.py --provider twelve --api-key YOUR_KEY analyze --ticker AAPL
```

### 2) M7 다중전략 앙상블 백테스트

기본 기간:
- 학습: 2015-01-01 ~ 2020-12-31
- 검증: 2021-01-01 ~ 2021-12-31
- 테스트: 2022-01-01 ~ 2024-12-31

```bash
python3 main.py --provider yahoo m7-ensemble
python3 main.py --provider twelve --api-key YOUR_KEY m7-ensemble --threshold 0.1
```

출력:
- 전략별 최적 파라미터/검증 성과
- 검증 성능 기반 전략 가중치
- 일 단위 매수/매도/관망 의견
- 주 단위 수익률 및 누적수익률

저장 파일 (`results/` 예시):
- `daily_opinion_<TICKER>.csv`
- `weekly_<TICKER>.csv`
- `leaderboard_<TICKER>_<STRATEGY>.csv`
- `weekly_all_m7.csv`

### 3) 미래 변동성 추정

```bash
python3 main.py --provider yahoo volatility --ticker AAPL --interval 1h --horizon-steps 24
python3 main.py --provider twelve --api-key YOUR_KEY volatility --ticker AAPL --interval 1day --horizon-steps 10
```

## 제약

- 머신러닝 라이브러리 미사용
- 시계열 순차 처리 (lookahead/data leakage 방지)
- 롱 포지션 1개만 보유
