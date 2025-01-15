import streamlit as st
import pandas as pd

st.markdown(
    '''
        <h2 style='text-align: center; color: #fff;'>
            Thyroid Cancer Recurrence Prediction
        </h2>
    ''', unsafe_allow_html=True
    )

st.markdown(
    '''
        - Find more projects here: [joaoaragao](%s)
        - Access github: [joaovictor-aragao](%s)
    ''' % ("https://joaovictor.onrender.com/", "https://github.com/joaovictor-aragao")
)

model_rf = st.session_state.ModelRF
df = st.session_state.dataset
df = df.drop(columns=['Age_cat'])
print(df.head())

con0 = st.container()

with con0:
    st.markdown(
        '''
            Insert the correspondent feature.
        '''
    )

    col0, col1, col2, col3 = st.columns(4)
    
    #--- first line
    with col0:
        v1 = st.number_input(label="Age")
    with col1:
        v2 = st.selectbox(
            label="Gender",
            options=df['Gender'].unique()
            )
    with col2:
        v3 = st.selectbox(
            label="Smoking",
            options=df['Smoking'].unique()
            )
    with col3:
        v4 = st.selectbox(
            label="Hx Smoking",
            options=df['Hx Smoking'].unique()
            )
        
    with col0:
        v5 = st.selectbox(
            label="Hx Radiothreapy",
            options=df['Hx Radiothreapy'].unique()
            )
    with col1:
        v6 = st.selectbox(
            label="Thyroid Function",
            options=df['Thyroid Function'].unique()
            )
    with col2:
        v7 = st.selectbox(
            label="Physical Examination",
            options=df['Physical Examination'].unique()
            )
    with col3:
        v8 = st.selectbox(
            label="Adenopathy",
            options=df['Adenopathy'].unique()
            )
        
    with col0:
        v9 = st.selectbox(
            label="Pathology",
            options=df['Pathology'].unique()
            )
    with col1:
        v10 = st.selectbox(
            label="Focality",
            options=df['Focality'].unique()
            )
    with col2:
        v11 = st.selectbox(
            label="Risk",
            options=df['Risk'].unique()
            )
    with col3:
        v12 = st.selectbox(
            label="T",
            options=df['T'].unique()
            )
        
    with col0:
        v13 = st.selectbox(
            label="N",
            options=df['N'].unique()
            )
    with col1:
        v14 = st.selectbox(
            label="M",
            options=df['M'].unique()
            )
    with col2:
        v15 = st.selectbox(
            label="Stage",
            options=df['Stage'].unique()
            )
    with col3:
        v16 = st.selectbox(
            label="Response",
            options=df['Response'].unique()
            )
        
    col4, col5, col6 , col7, col8 = st.columns(5)

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    with col6:
        st.button('Submit', on_click=click_button, type="secondary")

    if st.session_state.clicked:
        df.loc[len(df)] = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16, 'Yes']

        # create a categorical feature age: Age_cat
        interval = (10, 16, 30, 45, 120)
        cats = ['Children', 'Young Adults', 'Middle-Age Adults', 'Old Adults']
        df["Age_cat"] = pd.cut(df.Age, interval, labels=cats)
        
        for feat in list(df.columns):
            df[feat] = pd.factorize(df[feat])[0] + 1

        df = df.drop(columns='Recurred')

        y_pred_rf = model_rf.predict([
            df.loc[len(df)-1]
        ])

        if y_pred_rf[0] == 0:
            st.markdown(f'According to the model the results there is NO recurrency.')
        else:
            st.markdown(f'According to the model the results (unfortunally) there is recurrency.')


print('----')