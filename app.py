# 한국은행 금융안정보고서 RAG QA 챗봇 - Streamlit 앱
#
# 실행 방법
#   1) pip install streamlit langchain langchain_community langchain_openai langchain_text_splitters
#              langchain-classic chromadb openai tiktoken PyPDF2 pdfplumber -q
#   2) 이 파일(app.py)을 노트북(금융안정보고서_RAG_QA챗봇.ipynb)과 같은 Mini_Project 폴더에 둔다.
#   3) 터미널에서: streamlit run app.py
#   4) 사이드바에 그날의 OpenAI API 키를 입력하면 바로 사용 가능
#
# 특징
#   - 벡터 DB(Chroma)는 최초 1회만 PDF를 읽어 임베딩하고, 이후에는 디스크(chroma_db_fsr 폴더)에
#     저장된 것을 그대로 불러온다. API 키가 매일 바뀌어도 이미 만든 임베딩은 그대로 재사용 가능하다.
#     (임베딩을 만들 때만 유효한 키가 필요했고, 이후 조회는 그때그때 입력한 키로 하면 된다)
#   - 검색: 질문 분해(expand_query) + MMR 검색(multi_query_search) + LLM 재랭킹(rerank_relevance)
#   - 대화형: 이전 대화를 참고해 후속 질문("그거 더 자세히 알려줘" 등)을 독립 질문으로 재작성한 뒤 검색

import json
import os
import re
import shutil

import PyPDF2
import pdfplumber
import streamlit as st
import tiktoken
from openai import OpenAI

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_fsr")
COLLECTION_NAME = "fsr_2026_06_v2"

PDF_FILES = [
    "1. 개관 및 종합평가.pdf",
    "2. 금융안정상황.pdf",
    "4. 자영업 구조 변화에 따른 리스크 점검 및 대응방향.pdf",
    "현안1. 시장금리 상승에 따른 금융시스템 안정성 평가.pdf",
    "현안2. 비은행 부문 잠재리스크 점검 및 시사점.pdf",
    "참고3. 상업용부동산 시장 상황과 리스크 점검.pdf",
]

DISCLAIMER = (
    "\n\n---\n"
    "본 답변은 한국은행 금융안정보고서(2026년 6월)에 기초하며, "
    "투자/금융 결정의 법적 근거가 될 수 없습니다."
)

PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini, 100만 토큰당 USD
ENC = tiktoken.get_encoding("cl100k_base")

st.set_page_config(page_title="금융안정보고서 QA 챗봇", page_icon="💬")

# ---------------------------------------------------------------------------
# 사이드바: API 키 입력 (매일 바뀌는 학원 제공 키를 그때그때 입력)
# ---------------------------------------------------------------------------
st.sidebar.header("설정")

api_key_input = st.sidebar.text_input(
    "OpenAI API 키",
    type="password",
    help="저장되지 않고 이 브라우저 세션에서만 사용됩니다. 매일 새로 받은 키를 입력하세요.",
)

if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

api_ready = bool(os.getenv("OPENAI_API_KEY"))

st.title("한국은행 금융안정보고서 QA 챗봇")
st.caption("gpt-4o-mini 기반 RAG · 질문 분해 + MMR 검색 + 재랭킹 적용")

if not api_ready:
    st.info("왼쪽 사이드바에 OpenAI API 키를 입력하면 시작됩니다.")
    st.stop()


# ---------------------------------------------------------------------------
# PDF 전처리 함수 (노트북 Step2/Step3/12번과 동일한 로직)
# ---------------------------------------------------------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def mask_pii(text):
    text = re.sub(r"(\d{6})-\d{7}", r"\1-*******", text)
    text = re.sub(r"(\d{2,6}-\d{2,6})-\d{4,8}", lambda m: m.group(1) + "-" + "*" * 6, text)
    text = re.sub(r"(01[016789])-?\d{3,4}-?\d{4}", r"\1-****-****", text)
    return text


def extract_pdf_text(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_tables_as_markdown(path):
    md_tables = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            for t_idx, table in enumerate(page.extract_tables()):
                if not table or len(table) < 2:
                    continue
                header, rows = table[0], table[1:]

                def row_line(r):
                    return "| " + " | ".join((str(c).strip() if c else "") for c in r) + " |"

                lines = [row_line(header), "| " + " | ".join(["---"] * len(header)) + " |"]
                lines += [row_line(r) for r in rows]
                md_tables.append({"page": page_num, "table_idx": t_idx, "markdown": "\n".join(lines)})
    return md_tables


def build_documents():
    """PDF_FILES를 읽어 텍스트 청크 + 표 청크를 Document 리스트로 만든다."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_docs = []
    all_tables = {}

    for fname in PDF_FILES:
        path = os.path.join(BASE_DIR, fname)
        raw = extract_pdf_text(path)
        cleaned = mask_pii(clean_text(raw))
        for chunk in splitter.split_text(cleaned):
            text_docs.append(Document(page_content=chunk, metadata={"source": fname}))

        all_tables[fname] = extract_tables_as_markdown(path)

    table_docs = []
    for fname, tables in all_tables.items():
        for t in tables:
            md_text = f"[표 | {fname} {t['page']}페이지]\n{t['markdown']}"
            if len(md_text) <= 800:
                table_docs.append(
                    Document(page_content=md_text, metadata={"source": fname, "type": "table", "page": t["page"]})
                )
            else:
                lines = t["markdown"].split("\n")
                header_block = "\n".join(lines[:2])
                body_lines = lines[2:]
                chunk_lines, cur = [], []
                for line in body_lines:
                    cur.append(line)
                    if len("\n".join(cur)) > 500:
                        chunk_lines.append(cur)
                        cur = []
                if cur:
                    chunk_lines.append(cur)
                for cl in chunk_lines:
                    chunk_text = f"[표 | {fname} {t['page']}페이지]\n{header_block}\n" + "\n".join(cl)
                    table_docs.append(
                        Document(
                            page_content=chunk_text,
                            metadata={"source": fname, "type": "table", "page": t["page"]},
                        )
                    )

    return text_docs + table_docs


@st.cache_resource(show_spinner="벡터 DB를 준비하는 중입니다 (최초 1회만 PDF를 읽습니다)...")
def load_or_build_db(_api_key_marker: str):
    """디스크에 저장된 DB가 있으면 불러오고, 없으면 PDF부터 새로 만든다."""
    embeddings = OpenAIEmbeddings()

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    all_docs = build_documents()
    db = Chroma.from_documents(
        all_docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    db.persist()
    return db


# 사이드바: 벡터 DB 상태 표시 + 재구축 버튼
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    st.sidebar.success("벡터 DB: 저장된 데이터 사용")
else:
    st.sidebar.warning("벡터 DB: 최초 실행 - PDF를 새로 읽습니다")

if st.sidebar.button("벡터 DB 새로 만들기 (PDF 변경 시)"):
    load_or_build_db.clear()
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    st.rerun()

if st.sidebar.button("대화 초기화"):
    st.session_state.messages = []
    st.rerun()

db = load_or_build_db(os.getenv("OPENAI_API_KEY"))
st.sidebar.caption(f"저장된 조각 수: {db._collection.count()}개")


# ---------------------------------------------------------------------------
# LLM / RAG 파이프라인 (노트북 10, 11번과 동일한 로직)
# ---------------------------------------------------------------------------
client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPT = """너는 한국은행 금융안정보고서를 기반으로 답하는 거시 금융 리스크 분석 도우미다.
아래 [문서]에 있는 내용만 근거로 답하라.
여러 조각에 흩어져 있는 내용이라도 사실관계가 서로 어긋나지 않는다면 종합해서 요약해도 된다.
[문서] 전체를 살펴봐도 관련 내용이 전혀 없을 때만 "자료에서 확인할 수 없습니다"라고 답하라.
수치나 통계를 인용할 때는 문서에 있는 표현을 그대로 사용하라.

[문서]
{context}"""

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)


def expand_query(question, n=3):
    """복합 질문 -> 벡터 검색에 유리한 짧은 하위 검색어 n개"""
    prompt_text = f"""다음 질문에 답하기 위해 벡터DB에서 검색할 하위 검색어를 {n}개 만들어라.
각 검색어는 핵심 키워드 중심의 짧은 구/문장으로 작성하고, 한 줄에 하나씩만 출력하라.
번호, 설명, 따옴표는 붙이지 마라.

질문: {question}"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
    )
    lines = [l.strip("-• ").strip() for l in r.choices[0].message.content.split("\n") if l.strip()]
    return lines[:n]


def multi_query_search(question, k_per_query=4, fetch_k=20):
    """원 질문 + 하위 검색어들을 MMR로 각각 검색한 뒤 중복 제거해 병합"""
    sub_queries = expand_query(question)
    all_queries = [question] + sub_queries

    seen, merged_docs = set(), []
    for q in all_queries:
        hits = db.max_marginal_relevance_search(q, k=k_per_query, fetch_k=fetch_k, lambda_mult=0.5)
        for h in hits:
            key = h.page_content[:50]
            if key not in seen:
                seen.add(key)
                merged_docs.append(h)

    return merged_docs, sub_queries


def rerank_relevance(question, docs_, top_n=6):
    """후보 조각들을 질문과의 관련도 0~5점으로 한 번에 채점해 상위 top_n개만 남긴다"""
    if not docs_:
        return [], []

    numbered = "\n\n".join(f"[{i}] {d.page_content[:400]}" for i, d in enumerate(docs_))
    prompt_text = f"""아래는 질문과 검색된 문서 조각 목록이다.
각 조각이 질문에 답하는 데 얼마나 관련 있는지 0~5점으로 채점하라.
0: 전혀 관련 없음, 5: 질문에 직접 답할 수 있을 만큼 관련 높음.
반드시 JSON으로만 답하라. 형식: {{"scores": [조각0점수, 조각1점수, ...]}}

질문: {question}

[조각 목록]
{numbered}"""

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(r.choices[0].message.content)
    scores = result.get("scores", [0] * len(docs_))
    scores = (scores + [0] * len(docs_))[: len(docs_)]

    ranked = sorted(zip(scores, docs_), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]], scores


def condense_question(history, question):
    """대화 이력을 참고해 후속 질문을 독립적인 질문으로 재작성 (없으면 원 질문 그대로)"""
    if not history:
        return question

    history_text = "\n".join(
        f"{'사용자' if role == 'user' else '챗봇'}: {content}" for role, content in history[-6:]
    )
    prompt_text = f"""아래는 이전 대화와 사용자의 새 질문이다.
새 질문이 이전 대화를 알아야만 이해되는 후속 질문이면, 대화 맥락을 반영해 하나의 독립적인 질문으로 다시 써라.
이미 독립적으로 이해 가능한 질문이면 그대로 반환하라. 질문만 출력하고 다른 설명은 붙이지 마라.

[이전 대화]
{history_text}

[새 질문]
{question}"""

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
    )
    return r.choices[0].message.content.strip()


def estimate_cost(input_text, output_text):
    in_tok = len(ENC.encode(input_text))
    out_tok = len(ENC.encode(output_text))
    cost = (in_tok / 1e6) * PRICE_IN + (out_tok / 1e6) * PRICE_OUT
    return in_tok, out_tok, cost


def ask_v3(question, k_per_query=4, top_n=6):
    """질문 분해 + MMR 검색 + LLM 재랭킹 + 완화된 프롬프트를 적용한 답변 생성"""
    merged_docs, _ = multi_query_search(question, k_per_query=k_per_query)
    reranked_docs, _ = rerank_relevance(question, merged_docs, top_n=top_n)
    answer = documents_chain.invoke({"input": question, "context": reranked_docs})
    answer = answer + DISCLAIMER
    sources = sorted(set(d.metadata.get("source", "알수없음") for d in reranked_docs))
    return answer, sources, reranked_docs


# ---------------------------------------------------------------------------
# 채팅 UI (대화 기록 유지 + 후속 질문 재작성)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{'role': 'user'|'assistant', 'content': str, 'sources': [...]}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("출처: " + ", ".join(msg["sources"]))

user_question = st.chat_input("질문을 입력하세요")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    history_pairs = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중입니다..."):
            standalone_q = condense_question(history_pairs, user_question)
            answer, sources, reranked_docs = ask_v3(standalone_q)

            context_text = "\n".join(d.page_content for d in reranked_docs)
            in_tok, out_tok, cost = estimate_cost(context_text + standalone_q, answer)

            st.markdown(answer)
            if sources:
                st.caption("출처: " + ", ".join(sources))
            if standalone_q != user_question:
                st.caption(f"(대화 맥락을 반영해 재작성된 질문: {standalone_q})")
            st.caption(f"예상 비용: 입력 {in_tok} / 출력 {out_tok} 토큰 (${cost:.5f})")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
