"""Streamlit으로 실행하는 파일

[실행 방법]
1. 터미널 접속
2. 현재 폴더로 이동(cd)
3. streamlit run app.py


[참고사항]
Streamlit은 사용자가 화면(위젯)을 조작하게 되면 해당 파일(app.py)를 위에서 아래로 훑으며 다시 실행한다.
"""

import logging # 오류를 파이썬 터미널에 출력하는 라이브러리
import streamlit as st

from rag import (
SommelierConfigurationError, # 사용자 정의 예외
ai_sommelier_rag, # 이미지 -> 와인 추천 체인
validate_public_image_url # 이미지 url 검사
)

# 개발자가 보기 위한 로그를 출력하는 객체 생성
logger = logging.getLogger(__name__)

# 브라우저 탭 아이콘, 제목 설정, 화면 가운데 정렬
st.set_page_config(
    page_title = "AI Wine Sommelier",
    page_icon="🍷",
    layout="centered",
)

# 페이지 제목
st.title('🍷 AI Wine Sommelier')
st.write('음식 이미지 URL을 입력하면 Wine Magazine 리뷰를 검색해 어울리는 와인을 추천한다.')

st.info("공개 HTTPS 이미지 URL만 사용한다. 제출한 URL은 음식 분석을 위해 OpenAI에 전달된다.")

# st.form()은 여러 입력을 하나로 묶어서 제출할 수 있게 하는 역할
# st.form_submit_button() : 해당 form의 모든 입력을 제출하는 역할
with st.form(key = "image_url_form", clear_on_submit = False):

    image_url_input = st.text_input(
        '음식 이미지 URL',
        placeholder='https://example.com/food.jpg', # 힌트
        max_chars=2048,
        help='로그인 없이 직접 접근 가능한 HTTPS 이미지 주소 입력'
    )

    submitted = st.form_submit_button('와인 추천 받기', type='primary')

# submitted 버튼이 클릭되면 True, 아니면 False
if submitted:
    try:
        # 입력된 이미지 URL이 문제가 없는지 검증
        image_url = validate_public_image_url(image_url_input or "")

    except ValueError as error:
        st.warning(str(error))

    else: # image_url이 검증을 통과한 경우
        st.image(
            image_url,
            caption="와인 페어링을 요청한 음식 이미지",
            width='stretch'
        )

        st.subheader('AI 소믈리에 - 와인 추천')
        try:
            with st.spinner("음식과 관련된 와인 리뷰를 분석중입니다..."):
                st.write_stream(ai_sommelier_rag(image_url))

        # 설정 오류
        except SommelierConfigurationError as error:
            st.error(str(error))

        # 나머지 오류
        except Exception as error:
            logger.exception("[AI Sommelier Rag 실행 실패]")
            st.error(
                '추천 기능 수행 중 오류 발생'
                f'오류 타입 : {type(error).__name__}'
            )
            st.caption(
                "API KEY 확인, "
                "모델 접근 권한 확인,"
                "Pinecone 인덱스, 차원수, namespace 확인"
            )
