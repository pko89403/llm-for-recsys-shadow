import streamlit as st
from typing import Optional

def add_chat_message(role: str, message: str, avatar: Optional[str] = None):
    """채팅 기록에 채팅 메시지를 추가합니다.

    Args:
        role (str): 메시지 보낸 사람의 역할입니다.
        message (str): 메시지의 내용입니다.
        avatar (Optional[str], optional): 메시지 보낸 사람의 아바타입니다. 기본값은 None입니다.
    """
    if "chat_history" not in st.session_state:
        st.session.state.chat_history = []
    st.session_state.chat_history.append({"role": role, "message": message})
    if avatar is not None:
        st.chat_message(role, avatar=avatar).markdown(message)
    else:
        st.chat_message(role).markdown(message)

def get_color(agent_type: str) -> str:
    """주어진 에이전트 타입에 대한 색상을 반환합니다.

    Args:
        agent_type (str): 에이전트 타입입니다.

    Returns:
        str: 색상입니다.
    """
    if "manager" in agent_type.lower():
        return "rainbow"
    elif "reflector" in agent_type.lower():
        return "orange"
    elif "searcher" in agent_type.lower():
        return "blue"
    elif "interpreter" in agent_type.lower():
        return "green"
    elif "analyst" in agent_type.lower():
        return "red"
    else:
        return "gray"