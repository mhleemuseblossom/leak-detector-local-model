# 🛡️ SENTINEL — 정보 유출 탐지 시스템

HuggingFace MLX LM (Qwen2-1.5B) 기반 자동화 정보 유출 탐지 플랫폼

## 개요

로컬 MLX 모델을 활용하여 무료 불법 스트리밍/다운로드 사이트를 탐지합니다. **API 키 없이 로컬에서 동작합니다.**

---

## 프로젝트 구조

```
leak-detector/
├── backend/
│   ├── main.py           # FastAPI 서버
│   ├── crawler.py        # Playwright + httpx 크롤러
│   ├── hf_service.py     # MLX LM 모델 추론
│   └── database.py       # SQLite DB
├── frontend/
│   └── index.html        # 웹 대시보드
├── data/                 # DB 자동 생성 경로
├── requirements.txt
└── start.sh
```

---

## 동작 파이프라인

### 1단계: 키워드 생성
- **입력**: 탐지 주제 (예: "무료 만화")
- **처리**: MLX LM (Qwen2-1.5B) 프롬프트 기반 키워드 생성
- **출력**: 10-15개 검색 키워드

### 2단계: 웹 크롤링
- 키워드별 검색 및 페이지 수집
- Playwright로 JS 렌더링
- 최대 깊이 2, 키워드당 최대 5페이지

### 3단계: 텍스트 분석
- 각 페이지 텍스트 추출
- MLX LM으로 분석 (프롬프트 기반)
- 위험도 판정: HIGH / MEDIUM / LOW / NONE

### 4단계: 결과 저장
- SQLite DB에 저장
- 대시보드에 실시간 표시

---

## 기술 상세

### MLX LM 모델

| 항목 | 값 |
|------|-----|
| 모델 | `mlx-community/Qwen2-1.5B-Instruct` |
| 크기 | 약 3GB |
| 동작 | Apple Silicon (M1/M2/M3) |

### 분석 방법

#### 1. 프롬프트 기반 분석 (`analyze_with_prompt`)
```python
# MLX LM으로 텍스트 분석
prompt = """...
답변은 JSON 형식으로: {"risk_level": "HIGH|MEDIUM|LOW|NONE", ...}
"""

response = lm.generate(_model, _tokenizer, prompt, max_tokens=300)
```

#### 2. Similarity 기반 분석 (`analyze_text_for_leaks`)
- 사전 정의된 키워드와 텍스트 간 유사도 계산
- Falcon-7B 임베딩 사용

### 성능 최적화

**문제**: 매번 phrase마다 embedding 계산 → 느림 (53 forward/페이지)

**해결**: 시작 시 phrase embedding **미리 계산**

```python
# 초기화 시 1번만
for phrase, leak_type in LEAK_PHRASES:
    emb = _get_embedding(phrase)
    _leak_phrase_embeddings.append((emb, leak_type))

# 분석 시: 텍스트만 1회
text_emb = _get_embedding(combined_text)

# Similarity: dot product만
for phrase_emb, leak_type in _leak_phrase_embeddings:
    sim = dot(text_emb, phrase_emb)
```

**결과**: 30배 이상 속도 향상

---

## 탐지 키워드 카테고리

### 무료 시청 / 스트리밍
무료 보기, 무료 시청, 무료 감상, 무료 스트리밍, 바로보기, 지금보기, 다시보기, 풀영상, 풀버전 등

### 드라마 / 영화
최신영화 무료보기, 영화 무료 스트리밍, 드라마 다시보기, 드라마 전편 등

### 다운로드 / 토렌트
토렌트, 토렌트 다운로드, 마그넷, 파일 다운로드, 고화질 다운로드 등

### 화질 표시
HD, FHD, 4K, 1080p, 720p, 블루레이, BRRip, WEBRip 등

### 광고 / 회원 패턴
VIP 회원, 프리미엄 회원, 광고 후 시청, 로그인 후 시청 등

### 서버 패턴
서버1, 서버2, 스트리밍 서버, 플레이어, 대체 서버 등

---

## 실행 방법

### 1단계 — 의존성 설치

```bash
pip install -r requirements.txt
```

### 2단계 — MLX LM 설치

```bash
pip install mlx-lm
```

### 3단계 — Playwright 브라우저 설치

```bash
python -m playwright install chromium --with-deps
```

### 4단계 — 서버 실행

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

또는:
```bash
chmod +x start.sh && ./start.sh
```

### 5단계 — 브라우저에서 접속

```
http://localhost:8000
```

---

## 설정 조정 (crawler.py)

```python
self.max_depth = 2              # 링크 확산 깊이
self.max_pages_per_keyword = 5  # 키워드당 최대 페이지
```

---

## ⚠️ 주의

자사 정보 유출 모니터링 / 보안 연구 목적으로만 사용하세요.
