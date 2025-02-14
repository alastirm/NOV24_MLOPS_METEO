import streamlit as st

import emoji




st.set_page_config(page_title = 'MeteoStralia',
                   layout = 'wide',
                   page_icon = emoji.emojize('🦘'))


# css container
st.markdown("""
    <style>
    .container-margin {
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# css paragraphe
st.markdown("""
    <style>
    .paragraph {
        text-align: justify;
        color: rgb(16, 0, 97);
    }
    </style>
""", unsafe_allow_html=True)

# css image
st.markdown("""
    <style>
    .image {
        display: flex;
        justify-content: center;
        align-items: center;
        height : 5vh;
    }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title('MeteoStralia')

st.html("""
    <div class="header" style = 'margin-top : -180px; margin-bottom : 50px'>
        <h1 style = "color : rgb(16, 0, 97); font-size : 550%; height : 50px;">MeteoStralia </h1>
        <h4 style = 'color : rgb(156, 4, 4); margin-left : 50px'>Data Scientist Weather Forecast Australia </h4>
    </div>
""")

with st.container(border = False):

    content, image = st.columns([2, 1], border = False)

    with content:

        st.html("""
            <div class="column" style="display: flex;justify-content: center; text-align: justify; color: rgb(16, 0, 97); height: 100%;">
                <p>We have developed an innovative and powerful application designed to predict rainfall across Australia, leveraging a combination of accurate, real-time weather data and cutting-edge analysis algorithms. By processing vast amounts of meteorological information, our app ensures precise forecasting, particularly when it comes to predicting the risk of rainfall, allowing users to make well-informed decisions. The app empowers individuals with the ability to stay updated on local weather conditions, focusing on the likelihood of rain and any associated risks, giving Australians the tools they need to plan ahead.
                    With its user-friendly and intuitive interface, the app ensures that anyone, regardless of technical expertise, can effortlessly check for real-time weather updates, making it an indispensable tool for day-to-day life. Our primary goal is to enhance preparedness for individuals and communities by offering timely, reliable forecasts that provide clarity on weather patterns, particularly when facing unpredictable weather. By integrating advanced technology and expert meteorological insights, we aim to provide Australians with a solution that not only helps protect their daily activities but also enhances their ability to adapt and respond to weather challenges effectively. Through this fusion of technology and expertise, we’re striving to offer a practical, accessible tool for anyone in need of accurate weather information to navigate daily life.</p>
            </div>
        """)

    with image:
        st.markdown('<div class = "image">', unsafe_allow_html = True)
        st.image('image/australia.jpg', use_container_width = True)
        st.markdown('</div>', unsafe_allow_html = True)

with st.container(border = False):

    st.markdown('<div class="container-margin"></div>', unsafe_allow_html = True)

    image2, content2 = st.columns([1, 2], border = False)

    with image2:
        st.markdown('<div class = "image">', unsafe_allow_html = True)
        st.image('image/kangourou.png')
        st.markdown('</div>', unsafe_allow_html = True)

    with content2:

        st.html('''
            <div class="row" style="display: flex;justify-content: center; text-align: justify; color: rgb(16, 0, 97); height: 100%; margin-top : 50px">
                    <p>Meteostralia provides the most accurate and reliable rainfall forecasts available, making it the go-to site for anyone looking to stay ahead of the weather. Thanks to advanced meteorological technology and a team of expert analysts, we are able to deliver forecasts that are highly precise and up-to-date. Whether you're planning outdoor events, managing agricultural activities, or just preparing for your day-to-day routine, Meteostralia ensures you have the best information to make informed decisions. We use the latest satellite data, sophisticated models, and real-time updates to give you an unparalleled level of accuracy when it comes to predicting rainfall. Our commitment to quality means that every forecast is tailored to meet your specific location and needs, allowing you to trust that you're always prepared, no matter the weather. With Meteostralia, you'll never be caught off guard by the rain again.</p>
                </div>
            </div>
        ''')

with st.container(border = False):

    st.markdown('<div class="container-margin"></div>', unsafe_allow_html=True)

    content3, image3 = st.columns([2, 1], border = False)

    with content3 :
        st.html('''
            <div class="row" style="display: flex;justify-content: center; text-align: justify; color: rgb(16, 0, 97); height: 100%;">
            <p>The work we’ve undertaken has been incredibly demanding, with every step requiring immense focus and dedication. Our team of specialists, driven by their commitment to perfection, has worked tirelessly for hours on end, continuously pushing the boundaries of what’s possible. They dedicated themselves to finding the most precise settings, testing and retesting countless configurations to ensure no detail was overlooked. Each decision was carefully evaluated, and every adjustment was made with the utmost precision to fine-tune the system to its highest potential. This process involved not only technical expertise but also a deep understanding of the complexities involved, making the work even more challenging. Yet, through perseverance and collaboration, they were able to refine and optimize every aspect, working well beyond normal hours, driven by the desire to achieve the most accurate and reliable outcomes. Their efforts were truly relentless, with a clear goal in mind: to deliver a level of performance that exceeds expectations. It’s thanks to their unwavering dedication that we can confidently say we’ve found the best possible solutions after an extensive and demanding journey.</p>
            </div>
        ''')


    with image3:
        st.markdown('<div class = "image"></div>', unsafe_allow_html = True)
        st.image('image/orage.jpg')


# """)

# st.image('image/kangourou.png', caption='Kangourou Uluru', use_column_width=True)
