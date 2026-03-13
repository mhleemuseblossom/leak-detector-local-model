"""
hf_service.py - HuggingFace mlx-lm for leak detection using Qwen2-0.5B
"""
import mlx.core as mx
import mlx_lm as lm
import numpy as np
from typing import List, Dict

_model = None
_tokenizer = None
_model_loaded = False

LEAK_PHRASES_WITH_TYPES = [
    # 무료 시청 / 스트리밍
    ("무료 보기", "streaming"),
    ("무료 시청", "streaming"),
    ("무료 감상", "streaming"),
    ("무료 스트리밍", "streaming"),
    ("무료 다시보기", "streaming"),
    ("무료 영상", "streaming"),
    ("바로보기", "streaming"),
    ("지금보기", "streaming"),
    ("지금 시청", "streaming"),
    ("바로 재생", "streaming"),
    ("영상보기", "streaming"),
    ("전체보기", "streaming"),
    ("풀영상", "streaming"),
    ("풀버전", "streaming"),
    ("전편보기", "streaming"),
    ("전편 무료", "streaming"),
    ("다시보기", "streaming"),
    ("재생", "streaming"),
    
    # 최신 영화
    ("최신영화 무료보기", "movie"),
    ("최신영화 다시보기", "movie"),
    ("영화 무료보기", "movie"),
    ("영화 무료 스트리밍", "movie"),
    ("영화 다시보기", "movie"),
    ("영화 전편", "movie"),
    ("영화 풀버전", "movie"),
    ("영화 전체보기", "movie"),
    
    # 드라마
    ("드라마 다시보기", "drama"),
    ("드라마 무료보기", "drama"),
    ("드라마 무료 시청", "drama"),
    ("드라마 전편", "drama"),
    ("드라마 풀버전", "drama"),
    ("드라마 스트리밍", "drama"),
    
    # 애니
    ("애니 무료보기", "animation"),
    ("애니 다시보기", "animation"),
    ("애니 스트리밍", "animation"),
    ("애니 전편", "animation"),
    ("애니 풀버전", "animation"),
    
    # 다운로드 / 토렌트
    ("토렌트", "torrent"),
    ("토렌트 다운로드", "torrent"),
    ("마그넷", "torrent"),
    ("마그넷 링크", "torrent"),
    ("파일 다운로드", "torrent"),
    ("직접 다운로드", "torrent"),
    ("고화질 다운로드", "torrent"),
    ("1080p 다운로드", "torrent"),
    ("720p 다운로드", "torrent"),
    
    # 화질 표시
    ("고화질", "quality"),
    ("초고화질", "quality"),
    ("HD", "quality"),
    ("FHD", "quality"),
    ("4K", "quality"),
    ("1080p", "quality"),
    ("720p", "quality"),
    ("480p", "quality"),
    ("블루레이", "quality"),
    ("BRRip", "quality"),
    ("WEBRip", "quality"),
    ("HDRip", "quality"),
    ("CAM", "quality"),
    
    # 광고 / 회원 패턴
    ("VIP 회원", "ad"),
    ("프리미엄 회원", "ad"),
    ("광고 후 시청", "ad"),
    ("광고 건너뛰기", "ad"),
    ("광고 제거", "ad"),
    ("회원가입 후 시청", "ad"),
    ("로그인 후 시청", "ad"),
    
    # 서버 패턴
    ("서버1", "server"),
    ("서버2", "server"),
    ("서버3", "server"),
    ("스트리밍 서버", "server"),
    ("영상 서버", "server"),
    ("플레이어", "server"),
    ("영상 재생", "server"),
    ("대체 서버", "server"),
    ("백업 서버", "server"),
    ("재생 서버", "server"),
    
    # 짧은 형태
    ("무료", "streaming"),
    ("시청", "streaming"),
    ("영상", "streaming"),
    ("보기", "streaming"),
]

NON_LEAK_PHRASES = [
    ("official website", ""),
    ("licensed streaming", ""),
    ("subscription service", ""),
    ("buy now", ""),
    ("rent movie", ""),
    ("official trailer", ""),
    ("cast and crew", ""),
    ("release date", ""),
    ("box office", ""),
    ("critics review", ""),
    ("information about", ""),
    ("news article", ""),
    ("review", ""),
    ("discussion forum", ""),
    ("fan theory", ""),
    ("behind the scenes", ""),
    ("making of", ""),
    ("interview", ""),
]

HIGH_THRESHOLD = 0.65
MEDIUM_THRESHOLD = 0.35
LOW_THRESHOLD = 0.15

_leak_phrase_embeddings = []
_non_leak_phrase_embeddings = []
_embeddings_cached = False
_embedding_model = None


def _load_model():
    global _model, _tokenizer, _model_loaded
    
    if _model_loaded and _model is not None:
        return
    
    try:
        print("Loading mlx-community/Qwen2-0.5B-Instruct...")
        _model, _tokenizer = lm.load("mlx-community/Qwen2-0.5B-Instruct")
        _model_loaded = True
        print("Model loaded successfully!")
        _cache_phrase_embeddings()
    except Exception as e:
        print(f"Model load failed: {e}")


def _cache_phrase_embeddings():
    global _leak_phrase_embeddings, _non_leak_phrase_embeddings, _embeddings_cached
    if _embeddings_cached:
        return
    
    print("Caching phrase embeddings...")
    for phrase, leak_type in LEAK_PHRASES_WITH_TYPES:
        emb = _get_embedding(phrase)
        _leak_phrase_embeddings.append((emb, leak_type))
    
    for phrase, _ in NON_LEAK_PHRASES:
        emb = _get_embedding(phrase)
        _non_leak_phrase_embeddings.append(emb)
    
    _embeddings_cached = True
    print(f"Cached {len(_leak_phrase_embeddings)} leak + {len(_non_leak_phrase_embeddings)} non-leak phrases")


def _get_embedding(text: str) -> np.ndarray:
    _load_model()
    
    tokens = _tokenizer.encode(text)
    if not tokens:
        return np.zeros(384)
    
    input_ids = mx.array(tokens)
    
    if hasattr(_model, 'model') and hasattr(_model.model, 'embed_tokens'):
        token_embeddings = _model.model.embed_tokens(input_ids[None, :])
        embedding = mx.mean(token_embeddings, axis=1)[0]
    else:
        outputs = _model(input_ids[None, :])
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        embedding = mx.mean(hidden_states, axis=1)[0]
    
    embedding_np = np.array(embedding)
    embedding_norm = np.linalg.norm(embedding_np, axis=-1, keepdims=True)
    embedding_np = embedding_np / np.clip(embedding_norm, 1e-8, np.inf)
    return embedding_np


def _compute_similarity(text1: str, text2: str) -> float:
    emb1 = _get_embedding(text1)
    emb2 = _get_embedding(text2)
    return float(np.dot(emb1, emb2))


def generate_keywords(topic: str) -> List[str]:
    _load_model()
    
    topic_emb = _get_embedding(topic)
    
    similarities = []
    for i, (phrase_emb, leak_type) in enumerate(_leak_phrase_embeddings):
        sim = float(np.dot(topic_emb, phrase_emb))
        original_phrase = LEAK_PHRASES_WITH_TYPES[i][0]
        similarities.append((original_phrase, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    keywords = []
    seen = set()
    for phrase, sim in similarities:
        if sim > 0.3 and phrase not in seen:
            cleaned = phrase.strip()
            if len(cleaned) > 2:
                keywords.append(cleaned)
                seen.add(phrase)
        if len(keywords) >= 15:
            break
    
    if len(keywords) < 5:
        fallback_keywords = [
            f"{topic} 무료 보기",
            f"{topic} 스트리밍",
            f"{topic} 토렌트",
            f"{topic} 다시보기",
            f"{topic} 무료 시청"
        ]
        for kw in fallback_keywords:
            if kw not in seen and len(keywords) < 15:
                keywords.append(kw)
                seen.add(kw)
    
    return keywords[:15]


def generate_keywords_with_prompt(topic: str) -> List[str]:
    """프롬프트 기반 키워드 생성 (mlx-lm Qwen2-7B 사용)"""
    print(f"[generate_keywords_with_prompt] called with topic: {topic}")
    _load_model()
    print(f"[generate_keywords_with_prompt] model loaded, generating keywords...")
    
    prompt = f"""<|im_start|>system
당신은 웹 검색 전문가입니다. 다음 주제와 관련된 무료 미디어 공유 사이트를 찾기 위한 검색 키워드를 생성해주세요.

요구사항:
- 무료 스트리밍, 토렌트, 파일공유, 업로더 사이트 관련 키워드 포함
- 한국어/영어 혼합 가능
- 드라마, 영화, 애니메이션, 음악 등 다양한 콘텐츠 유형 고려
- 1개 키워드 생성

답변은 반드시 아래 JSON 형식으로만 응답하세요:
{{"keywords": ["키워드1", "키워드2", ...]}}
<|im_end|>
<|im_start|>user
주제: {topic}
<|im_end|>
<|im_start|>assistant
"""
    
    try:
        response = lm.generate(_model, _tokenizer, prompt, max_tokens=300)
        output = response.strip()
        
        import re
        import json
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            keywords = data.get("keywords", [])
            if keywords:
                return keywords[:15]
    except Exception as e:
        print(f"[MLX 키워드 생성 오류]: {e}")
    
    return generate_keywords(topic)


def analyze_text_for_leaks(url: str, title: str, text: str) -> Dict:
    if not text or len(text.strip()) < 50:
        return {
            "risk_level": "NONE",
            "risk_score": 0.0,
            "leak_types": [],
            "summary": "분석할 텍스트 없음"
        }
    
    _load_model()
    
    combined_text = f"{title} {text}"[:2000]
    text_emb = _get_embedding(combined_text)
    
    leak_similarities = []
    for phrase_emb, leak_type in _leak_phrase_embeddings:
        sim = float(np.dot(text_emb, phrase_emb))
        leak_similarities.append((sim, leak_type))
    
    non_leak_similarities = []
    for phrase_emb in _non_leak_phrase_embeddings:
        sim = float(np.dot(text_emb, phrase_emb))
        non_leak_similarities.append(sim)
    
    avg_leak_sim = np.mean([s[0] for s in leak_similarities]) if leak_similarities else 0.0
    avg_non_leak_sim = np.mean(non_leak_similarities) if non_leak_similarities else 0.0
    
    if avg_leak_sim + avg_non_leak_sim < 1e-8:
        risk_score = 0.0
    else:
        risk_score = avg_leak_sim / (avg_leak_sim + avg_non_leak_sim)
    
    if risk_score >= HIGH_THRESHOLD:
        risk_level = "HIGH"
    elif risk_score >= MEDIUM_THRESHOLD:
        risk_level = "MEDIUM"
    elif risk_score >= LOW_THRESHOLD:
        risk_level = "LOW"
    else:
        risk_level = "NONE"
    
    leak_types = []
    threshold_for_type = 0.4
    sorted_leak_sim = sorted(leak_similarities, key=lambda x: x[0], reverse=True)
    seen_types = set()
    for sim, leak_type in sorted_leak_sim:
        if sim >= threshold_for_type and leak_type not in seen_types:
            leak_types.append(leak_type)
            seen_types.add(leak_type)
        if len(leak_types) >= 3:
            break
    
    if risk_level == "NONE":
        summary = "무료 미디어 콘텐츠가 감지되지 않았습니다."
    elif risk_level == "LOW":
        summary = "무료 미디어 콘텐츠와 약간의 유사성이 감지되었습니다."
    elif risk_level == "MEDIUM":
        summary = f"무료 미디어 콘텐츠 유사성이 감지되었습니다. 유형: {', '.join(leak_types)}"
    else:
        summary = f"명백한 무료 미디어 콘텐츠 유사성이 감지되었습니다. 유형: {', '.join(leak_types)}"
    
    if risk_level == "HIGH" and url:
        summary += f" URL: {url[:50]}..."
    
    return {
        "risk_level": risk_level,
        "risk_score": float(risk_score),
        "leak_types": leak_types,
        "summary": summary
    }


def analyze_with_prompt(url: str, title: str, text: str) -> Dict:
    """프롬프트 기반 분석 (mlx-lm Qwen2-7B 사용)"""
    if not text or len(text.strip()) < 50:
        return {
            "risk_level": "NONE",
            "risk_score": 0.0,
            "leak_types": [],
            "summary": "분석할 텍스트 없음"
        }
    
    _load_model()
    
    truncated_text = text[:3000] if len(text) > 3000 else text
    
    prompt = f"""<|im_start|>system
당신은 웹 콘텐츠 분석 전문가입니다. 아래 웹페이지 텍스트를 분석하여 불법 무료 공유 사이트인지 판단해주세요.

분석 기준 (다음 중 하나라도 있으면 유출로 판단):
1. 영화/드라마/애니 무료 스트리밍 직접 제공 (다운로드, 보기 버튼)
2. 토렌트, 파일공유 링크 제공
3. 가입 없이 바로 시청 가능
4. 최신 작품 무료 제공

정식 플랫폼은 제외 (유출 아님):
- 넷플릭스, 디즈니+, 웨이브, 티빙, 쿠팡플레이, 왓챠 등 유료 플랫폼 공식사이트
- 유튜브 무료 공개 영상
- 영화사/제작사 공식 사이트

판단 기준:
- HIGH: 불법 무료 공유 명확
- MEDIUM: 의심 - 광고 많음, 가입 유도
- LOW: 관련 콘텐츠 가볍게 언급
- NONE: 불법 무료 공유 없음

답변은 반드시 아래 JSON 형식으로만 응답하세요:
{{"risk_level": "HIGH|MEDIUM|LOW|NONE", "risk_score": 0.0~1.0, "leak_types": ["유출유형1"], "reason": "분석 이유 (어떤 키워드/문구때문에 판단했는지)", "summary": "전체 요약"}}<|im_end|>
<|im_start|>user
URL: {url}
제목: {title}
텍스트: {truncated_text}<|im_end|>
<|im_start|>assistant
"""
    
    try:
        response = lm.generate(_model, _tokenizer, prompt, max_tokens=300)
        output = response.strip()
        
        import re
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            import json
            data = json.loads(json_match.group())
            return {
                "risk_level": data.get("risk_level", "NONE"),
                "risk_score": float(data.get("risk_score", 0.0)),
                "leak_types": data.get("leak_types", []),
                "reason": data.get("reason", ""),
                "summary": data.get("summary", "분석 완료")
            }
    except Exception as e:
        print(f"[MLX 프롬프트 분석 오류] {e}")
    
    return analyze_text_for_leaks(url, title, text)
