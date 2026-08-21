# VOC 주간 리포트 Streamlit 앱

이 앱은 `05_voc_weekly_report_practice.ipynb`의 고정 Multi-Agent workflow를 한 화면에서 실행한다. CSV의 수치는 pandas가 계산하고, 분석 Worker와 작성 Worker는 해석과 문장 생성을 담당한다.

```text
VOC CSV → KPI 계산 → 분석 Worker → 작성 Worker → Markdown·PDF
```

## 화면에서 하는 일

1. 15건의 기본 샘플이나 120건의 확장 샘플을 받아 화면에 업로드한다.
2. 업로드가 끝나면 전체 문의·해결률·평균 해결 시간·만족도와 미해결 문의를 확인한다.
3. `주간 보고서 만들기`를 눌러 두 Worker와 LangGraph를 실행한다.
4. 생성된 보고서를 화면에서 확인하고 Markdown 또는 PDF로 받는다.

## 파일 구성

- `app.py`: CSV 입력, KPI 카드, 보고서 생성 버튼과 다운로드 화면을 담당한다.
- `voc_report.py`: 데이터 검증, KPI 계산, 두 Agent Worker와 LangGraph를 담당한다.
- `data/support_tickets.csv`: 노트북과 같은 15건의 기본 샘플 데이터이다.
- `data/support_tickets_large.csv`: 한 주 동안 접수된 120건의 수업용 확장 데이터이다.
- `.env.example`: 필요한 환경 변수 이름만 보여 주는 예시이다.
- `requirements.txt`: 실행 패키지 범위를 기록한다.

## 설치와 실행

PyCharm Terminal에서 이 폴더로 이동한 뒤 패키지를 설치한다.

```bash
python -m pip install -r requirements.txt
```

`.env.example`을 참고해 이 폴더 또는 상위 폴더의 `.env`에 실제 key를 저장한다. key는 코드와 Git에 넣지 않는다.

```dotenv
OPENAI_API_KEY=발급받은_OpenAI_API_Key
OPENAI_CHAT_MODEL=gpt-5.6-luna
```

앱을 실행한다.

```bash
python -m streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 Terminal에 표시된 `http://localhost:8501` 주소로 접속한다. 앱을 종료할 때는 Terminal에서 `Ctrl+C`를 누른다.

## CSV 입력 계약

CSV에는 다음 열이 필요하다.

| 열 | 의미 |
| --- | --- |
| `ticket_id` | 문의 식별자 |
| `received_date` | 접수일 |
| `category` | 문의 분류 |
| `priority` | `high`, `medium`, `low` |
| `status` | `resolved`, `open` |
| `resolution_hours` | 해결된 문의의 처리 시간 |
| `satisfaction` | 만족도 `1`~`5` 또는 빈 값 |
| `summary` | 문의 요약 |

업로드 파일은 UTF-8 CSV, 2MB 이하를 사용한다. `resolved` 문의에는 `resolution_hours`가 필요하다.
CSV를 업로드하기 전에는 KPI·미해결 문의·보고서 생성 영역을 표시하지 않는다.

## 수업에서 확인할 경계

- KPI와 티켓 ID는 LLM이 아니라 pandas 결과에서 가져온다.
- 분석 Worker는 `get_support_metrics` Tool을 반드시 호출한다.
- 작성 Worker는 숫자 없는 정성 문장만 만든다.
- 문의 요약은 분석 데이터로만 취급하고 그 안의 명령을 따르지 않도록 역할 정책을 둔다.
- Streamlit form은 버튼을 누른 시점에만 모델을 호출한다.
- 버튼을 누르면 KPI와 미해결 문의 요약이 OpenAI에 전달되고 호출 비용이 발생한다. 실제 고객 개인정보를 넣지 않는다.

## 자주 발생하는 오류

- `OPENAI_API_KEY를 설정` 안내: `.env`의 변수 이름을 확인하고 Streamlit을 다시 시작한다.
- 패키지 설치 안내: 실행 중인 PyCharm interpreter에서 `requirements.txt`를 설치한다.
- PDF font 안내: 운영체제에 맑은 고딕, AppleGothic, 나눔고딕 또는 Noto Sans CJK를 설치한다. Markdown 다운로드는 계속 사용할 수 있다.
- 숫자 없는 문장 계약 오류: 같은 데이터로 보고서 생성 버튼을 다시 누른다.
