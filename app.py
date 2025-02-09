import streamlit as st


with st.sidebar:
    st.title('MeteoStralia')

    if st.button("Home"):
        st.switch_page("app.py")
    if st.button("By Location"):
        st.switch_page("pages/page_location.py")
    # if st.button("Page 2"):
    #     st.switch_page("pages/page_2.py")


st.title('MeteoStralia')
