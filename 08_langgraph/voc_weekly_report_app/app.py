"""VOC 주간 보고서 Multi-Agent workflow를 실행하는 간결한 Streamlit 화면이다."""

from __future__ import annotations

import hashlib
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from voc_report import (
    ReportConfigurationError,
    ReportGenerationError,
    TicketDataError,
    calculate_support_metrics,
    run_report_pipeline,
    validate_ticket_data,
)


APP_DIRECTORY = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = APP_DIRECTORY / "data" / "support_tickets.csv"
LARGE_SAMPLE_DATA_PATH = APP_DIRECTORY / "data" / "support_tickets_large.csv"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_REQUEST = "이번 주 고객센터 VOC와 운영 KPI를 근거로 주간 보고서를 작성해 줘."

logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="VOC 주간 리포트",
    page_icon="📋",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp { background: #f6f8fb; }
        .block-container { max-width: 980px; padding-top: 2.25rem; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6eaf0;
            border-radius: 14px;
            padding: 1rem 1.1rem;
        }
        [data-testid="stMetricLabel"] { color: #667085; }
        .stButton > button, .stDownloadButton > button {
            min-height: 2.75rem;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_sample_csv_bytes(path: str, modified_time_ns: int) -> bytes:
    """다운로드용 샘플 CSV bytes를 파일 수정 시점 기준으로 캐시한다."""
    del modified_time_ns
    return Path(path).read_bytes()


def read_uploaded_csv(uploaded_file) -> tuple[pd.DataFrame, bytes]:
    """업로드 크기를 제한하고 UTF-8 CSV를 DataFrame으로 읽는다."""
    csv_bytes = uploaded_file.getvalue()
    if len(csv_bytes) > MAX_UPLOAD_BYTES:
        raise TicketDataError("CSV 파일은 20MB 이하만 사용할 수 있습니다.")
    try:
        return pd.read_csv(BytesIO(csv_bytes)), csv_bytes
    except UnicodeDecodeError as error:
        raise TicketDataError("CSV 파일을 UTF-8로 저장한 뒤 다시 업로드 해주세요.") from error
    except pd.errors.ParserError as error:
        raise TicketDataError("CSV 행과 열 구분을 확인해주세요.") from error


st.title("VOC 주간 리포트")
st.caption("고객 문의 CSV에서 KPI를 계산하고 두 Worker가 보고서를 작성합니다.")

uploaded_file = st.file_uploader(
    "VOC CSV",
    type=["csv"],
    help="UTF-8 CSV를 업로드하면 KPI와 보고서 생성 화면이 열립니다.",
)

if uploaded_file is None:
    st.info("VOC CSV를 업로드하면 KPI와 미해결 문의 분석을 시작합니다.")
    sample_columns = st.columns(2)
    for column, label, path in (
        (sample_columns[0], "기본 샘플 받기 · 15건", SAMPLE_DATA_PATH),
        (sample_columns[1], "확장 샘플 받기 · 120건", LARGE_SAMPLE_DATA_PATH),
    ):
        try:
            sample_csv_bytes = read_sample_csv_bytes(
                str(path), path.stat().st_mtime_ns
            )
        except OSError:
            continue
        column.download_button(
            label,
            data=sample_csv_bytes,
            file_name=path.name,
            mime="text/csv",
            width="stretch",
        )
    st.stop()

try:
    raw_tickets, csv_bytes = read_uploaded_csv(uploaded_file)
    data_label = uploaded_file.name

    tickets = validate_ticket_data(raw_tickets)
    metrics = calculate_support_metrics(tickets)
except (OSError, TicketDataError) as error:
    st.error(str(error))
    st.stop()

data_fingerprint = hashlib.sha256(csv_bytes).hexdigest()
st.caption(f"현재 데이터: {data_label} · {metrics['period']}")

st.subheader("이번 주 현황")
metric_columns = st.columns(4)
metric_columns[0].metric("전체 문의", f"{metrics['total_tickets']}건")
metric_columns[1].metric("해결률", f"{metrics['resolved_rate_pct']}%")
metric_columns[2].metric(
    "평균 해결 시간", f"{metrics['avg_resolution_hours']}시간"
)
metric_columns[3].metric("평균 만족도", f"{metrics['avg_satisfaction']} / 5")

category_text = " · ".join(
    f"{name} {count}건" for name, count in metrics["category_counts"].items()
)
st.caption(
    f"SLA 초과 {metrics['sla_violation_count']}건 · "
    f"high 우선순위 미해결 {metrics['high_priority_open']}건 · {category_text}"
)

if metrics["open_examples"]:
    st.subheader("확인이 필요한 문의")
    open_cases = pd.DataFrame(metrics["open_examples"]).rename(
        columns={
            "ticket_id": "티켓",
            "category": "분류",
            "priority": "우선순위",
            "summary": "문의 요약",
        }
    )
    st.dataframe(open_cases, hide_index=True, width="stretch")

with st.expander("원본 데이터 보기"):
    st.dataframe(tickets, hide_index=True, width="stretch")

st.divider()
st.subheader("AI 보고서 생성")
st.caption(
    "버튼을 누르면 KPI와 미해결 문의 요약이 OpenAI에 전달되고 모델 호출 비용이 발생한다."
)

with st.form("report_form", clear_on_submit=False):
    request = st.text_area(
        "보고서 요청",
        value=DEFAULT_REQUEST,
        height=90,
        max_chars=500,
    )
    submitted = st.form_submit_button(
        "주간 보고서 만들기",
        type="primary",
        width="stretch",
    )

if submitted:
    st.session_state.pop("voc_report_result", None)
    st.session_state.pop("voc_report_fingerprint", None)
    try:
        with st.spinner("분석 Worker와 작성 Worker가 보고서를 만들고 있습니다..."):
            result = run_report_pipeline(tickets, request)
    except (TicketDataError, ReportConfigurationError, ReportGenerationError) as error:
        st.error(str(error))
    except Exception as error:  # noqa: BLE001 - 외부 SDK 원문은 화면에 노출하지 않는다.
        logger.exception("VOC 보고서 생성 중 예상하지 못한 오류가 발생했습니다")
        st.error(f"보고서 생성에 실패했다. 오류 유형: {type(error).__name__}")
        st.caption("PyCharm Terminal에서 모델 접근 권한과 네트워크 상태를 확인하세요.")
    else:
        st.session_state["voc_report_result"] = result
        st.session_state["voc_report_fingerprint"] = data_fingerprint

result = st.session_state.get("voc_report_result")
result_fingerprint = st.session_state.get("voc_report_fingerprint")

if result and result_fingerprint == data_fingerprint:
    st.success("KPI Tool 사용과 보고서 구조 검증을 통과했습니다.")
    st.caption(
        f"모델: {result['model_name']} · "
        "분석 Worker → 작성 Worker → 저장 Node"
    )

    st.subheader("생성된 보고서")
    st.markdown(result["report_markdown"])

    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Markdown 받기",
        data=result["report_markdown"].encode("utf-8"),
        file_name="voc_weekly_report.md",
        mime="text/markdown",
        width="stretch",
    )
    if result["pdf_bytes"]:
        download_columns[1].download_button(
            "PDF 받기",
            data=result["pdf_bytes"],
            file_name="voc_weekly_report.pdf",
            mime="application/pdf",
            width="stretch",
        )
    else:
        download_columns[1].button(
            "PDF를 만들 수 없습니다.",
            disabled=True,
            width="stretch",
        )
        st.warning(result["pdf_warning"])
elif result and result_fingerprint != data_fingerprint:
    st.info("데이터가 변경되어 새 데이터로 보고서를 다시 생성합니다.")
