import streamlit as st
from source.functions import create_barplots as cb

PLOT_COLORS = ["#3288BD", "#99D594", "#E6F598", "#FEE08B", "#FC8D59", "#D53E4F"]

### Page config
st.set_page_config(
    page_title='Differentiated Thyroid Cancer Recurrence EDA Project',
    page_icon="📈",
    layout='wide',
)

st.title("Differentiated Thyroid Cancer Recurrence EDA Project")
st.markdown(':first_place_medal: I am utilizing the Scrum methodology in this project, organizing work into story and tasks to ensure iterative progress.',
    unsafe_allow_html=True)

st.markdown(
    '''
        - Find more projects here: [joaoaragao](%s)
        - Access github: [joaovictor-aragao](%s)
    ''' % ("https://joaovictor.onrender.com/", "https://github.com/joaovictor-aragao")
)

st.markdown(
    """
    1. Story: Differentiated Thyroid Cancer Recurrence \\
        Tasks to complete the Story

        1. Import dataset: https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence
        2. Do proper EDA of the dataset and generate a report
        3. Do necessary preprocessing steps
        4. Perform a regression and classification model
        5. Compare those models using some KPI's

    2. Data understanding and collect
        1. The dataset used in this project is the Differentiated Thyroid Cancer Recurrence which contains 13 clinicopathologic features aiming to predict recurrence of well differentiated thyroid cancer. 
        The data set was collected in duration of 15 years and each patient was followed for at least 10 years.
        2. The models were used to solve the problem of predict differentiated thyroid cancer.
        
    """
)

con0 = st.container()
con1 = st.container()
con2 = st.container()

######### BRIEFING
with con0:
    st.header("Data exploration")
    st.markdown(
        """
            In this step, we will perform Exploratory Data Analysis (EDA) to extract insights from the dataset, 
            focusing on identifying the features that play a significant role in predicting Differentiated Thyroid Cancer Recurrence. 
            This involves conducting data analysis using Pandas and creating visualizations with Plotly. 
            Understanding the dataset thoroughly and uncovering meaningful insights is always a crucial first step in any analysis process.
        """
    )

    ## ---- 1
    st.markdown('1. IMPORT LIBRARIES')
    with st.echo():
        
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from warnings import filterwarnings
        filterwarnings('ignore')

    ## ---- 2
    st.markdown('2. LOAD DATA FROM FOLDER')
    with st.echo():
        
        df = pd.read_csv("dataset/Thyroid_Diff.csv")


    st.markdown('The first few rows of the DataFrame will be displayed for a quick preview of the data.')
    st.dataframe(df.head(5))

    st.markdown(
        '''
            The dataset repository does not provide a description of the features, so the following descriptions were created by me for clarity.
            - Age (continuous)
            - Gender (categorial)
            - Smoking (boolean)
            - Hx Smoking (boolean): historical of smoking
            - Hx Radiothreapy (boolean): historical fo radiothreapy
            - Thyroid Funtion (categorical): terms to specufy the amount of thyroid hormone
            - Physical Examination (categorical): a comprehensive assessment of a person's health status
            - Adenopathy (boolean): medical term that describes swollen lymph nodes
            - Pathology (categorical): identify pathology
            - Focality (categorical)
            - Risk (categorical)
            - T (categorial): tumor
            - N (categorial): nodule
            - M (categorial): metastasis
            - Stage (categorial)
            - Response (categorial)
            - Recurred (boolean): target

            However, some dataset information was shared in this [repo](%s) and is presented below as follows:
            - For what purpose was the dataset created? It was a part of research in the field of AI and Medicine
            - Who funded the creation of the dataset? No funding was provided.
            - What do the instances in this dataset represent? Individual patients
            - Are there recommended data splits? No
            - Does the dataset contain data that might be considered sensitive in any way? No
            - Has Missing Values? No
        ''' % "https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence"
    )

######### END BRIEFING

fig = go.Figure(
    data=[go.Pie(
        labels=df['Recurred'].value_counts().index, 
        values=df['Recurred'].value_counts().values, 
        textinfo='label+percent',
        insidetextorientation='radial',
        marker_colors = PLOT_COLORS[:len(df['Recurred'].value_counts().values)],
        title="Thyroid Cancer Recurred? (TARGET)"
        )])

fig.update(layout_showlegend=False)

######### CHARTS & TABLES
with con1:

    st.markdown(
        '''
            3. EXPLORATORY DATA ANALYSIS (EDA)

            In this section, you will find various charts and tables that provide a comprehensive overview of the dataset, highlighting key patterns, trends, and relationships to aid in understanding the data. 
        '''
    )

    st.markdown(
        '''
            FEATURE: RECURRED CANCER (TARGET)
        '''
    )

    st.plotly_chart(fig)

    st.markdown(
        '''
            In the plot above, is possible to see the distribution of thyroid cancer recurency. 
            In some steps foward, the models developed are going to predict and analyze thyroid cancer recurrence, which serves as the TARGET variable in this study.
        '''
    )

    st.markdown(
        '''
            FEATURE: AGE
        '''
    )

    fig_col1, fig_col2, fig_col3 = st.columns(3)

    with fig_col1:
        fig1 =  px.histogram(
            x=df.loc[df["Recurred"] == 'Yes']['Age'].values.tolist(),
            labels={'x':'Age', 'y':'Count'},
            color_discrete_sequence=[PLOT_COLORS[0]],
            title='Recurred'
        )

        fig1.update_layout(bargap=0.1)

        st.write(fig1)
    
    with fig_col2:
        fig2 =  px.histogram(
            x=df.loc[df["Recurred"] == 'No']['Age'].values.tolist(),
            labels={'x':'Age', 'y':''},
            color_discrete_sequence=[PLOT_COLORS[1]],
            title='No Recurred'
        )

        fig2.update_layout(bargap=0.1)

        st.write(fig2)

    with fig_col3:
        fig3 =  px.histogram(
            x=df['Age'].values.tolist(),
            labels={'x':'Age', 'y':'Count'},
            color_discrete_sequence=[PLOT_COLORS[2]],
            title='Overall Age'
        )

        fig3.update_layout(bargap=0.1)

        st.write(fig3)

    st.markdown(
        '''
            - On the recurred plot the distribution is bimodal, with two peaks around 40 years with the highest count (almost 30 cases) and a smaller around 60 years. 
            The number of recurred cancer cases drops significantly after 70 years, and only a few cases in individuals below 20 years.
            
            - On the second plot, the distribution is unimodal with a clear peak on 30 years (around 50 cases).
            The number of cases decreases steadily with age after 40 years. There are fewer cases above 70 years, but the drop is more gradual compared to the "Recurred Cancer" group.
            
            - The distribution is similar to the "No Recurred Cancer" chart, as it dominates the dataset. 
            The peak occurs at 30 years with a significant count (around 70 cases).

            Implications:

            - The age of 30–40 years might represent a critical period for increased cancer activity.
            - Preventative measures or monitoring might be particularly important in this age group to mitigate the risk of recurrence.
        '''
    )

    st.markdown('Now, let\'s create a categorical variable to handle with the Age.')

    # create a categorical feature age: Age_cat
    interval = (10, 16, 30, 45, 120)
    cats = ['Children', 'Young Adults', 'Middle-Age Adults', 'Old Adults']
    df["Age_cat"] = pd.cut(df.Age, interval, labels=cats)
    
    # First plot
    trace0 = go.Bar(
        x = df[df["Recurred"]== 'No']["Age_cat"].value_counts().sort_index().index.values,
        y = df[df["Recurred"]== 'No']["Age_cat"].value_counts().sort_index().values,
        name='No recurred',
        marker_color=PLOT_COLORS[0]
    )

    # Second plot
    trace1 = go.Bar(
        x = df[df["Recurred"]== 'Yes']["Age_cat"].value_counts().sort_index().index.values,
        y = df[df["Recurred"]== 'Yes']["Age_cat"].value_counts().sort_index().values,
        name="Recurred",
        marker_color=PLOT_COLORS[1]
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title='Age categories Distribuition'
    )

    fig4 = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig4)
    
    st.markdown(
        '''
            Insights from the plot above:
            - Children (10-16 years) have negligible cases, indicating lower prevalence of cancer or recurrence in this age group.
            - Young Adults (17-30 years) have fewer recurrence cases, likely reflecting a stronger recovery or better outcomes in this group.
            - Middle-Age Adults (31-45) experience the highest overall cancer cases, indicating this is a critical age for both recurrence and non-recurrence.
            - Old Adults (46+) show a higher proportion of recurrence relative to non-recurrence compared to younger groups, suggesting an increased likelihood of recurrence with age.
            
        '''
    )
    
    fig5 = px.box(
        df, x="Gender", y="Age", color="Recurred",
        labels={"Gender":"Gender"},
        color_discrete_sequence=PLOT_COLORS[:2]
        )

    st.plotly_chart(fig5)
    
    st.markdown(
        '''
            Insights from the Plot:

            1. **Age Distribution by Gender and Recurrence Status:**
            - **Non-Recurred Cases:**
                - Median age for both females (F) and males (M) is around **40 years**.
                - The interquartile range (IQR) is narrower for females compared to males, suggesting less variation in age among non-recurred cases in females.
                - There are several **outliers** in older ages for females (above 70 years), indicating some cases of non-recurrence in elderly women.

            - **Recurred Cases:**
                - Median age for both females (F) and males (M) is similar and slightly higher than for non-recurred cases, around **50 years**.
                - The IQR is wider for both genders compared to non-recurred cases, indicating greater variation in ages among recurrence cases.
                - No significant outliers are observed in the recurred cases for either gender.

            2. **Comparison Between Genders:**
            - For both recurrence and non-recurrence, the age distribution patterns for males and females are comparable, with no substantial differences in medians or IQRs.

            3. **Overall Trend:**
            - Recurred cases tend to occur at **older ages** compared to non-recurred cases.
            - Non-recurrence shows tighter clustering around younger ages, particularly for females.
        '''
    )

    st.markdown('Other features by target variable')

    DEFAULT_VALUE = ' - '.join(df.columns[2:5])
    CHOICES = pd.DataFrame({
        'Features' : [
            ' - '.join(df.columns[2:5]),
            ' - '.join(df.columns[5:8]),
            ' - '.join(df.columns[8:11]),
            ' - '.join(df.columns[11:14]),
            ' - '.join(df.columns[14:17])
        ],
        'Values' : [
            [2,3,4], 
            [5,6,7], 
            [8,9,10], 
            [11,12,13], 
            [14,15,16]
        ]
    })

    selected = st.selectbox(
        "Select three features to cross with Target:", 
        CHOICES['Features']
        )

    results = CHOICES['Values'][CHOICES[CHOICES['Features'] == selected].index[0]]
    
    fig_col6, fig_col7, fig_col8 = st.columns(3)

    with fig_col6:
        st.plotly_chart(cb(df, df.columns[results[0]]))
    with fig_col7:
        st.plotly_chart(cb(df, df.columns[results[1]]))
    with fig_col8:
        st.plotly_chart(cb(df, df.columns[results[2]]))

    with st.echo():

        df1 = df.copy() # make a dataframe copy

        # convert factor to integer
        for feat in list(df1.columns):
            df1[feat] = pd.factorize(df[feat])[0] + 1

        df1['Recurred'] = df1['Recurred'] - 1

    corr = round(df1.corr(), 4)

    fig9 = px.imshow(corr, text_auto=True, aspect="auto")

    # fig9 = px.imshow(corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '0pt'}))
    st.plotly_chart(fig9)

    st.markdown(
        '''
            Insights from correlation plot to the target feature
            - **Tumor size (T), Node involvement (N), Stage, and Risk** show the strongest correlations with recurrence, making them key factors in predicting cancer recurrence.
            - Demographic variables like **Gender** and lifestyle factors like **Smoking** exhibit negligible impact on recurrence.
            - Recurrence is slightly more likely in **younger patients** and those with **higher cancer staging or metastasis**.
        '''
    )

with con2:

    st.markdown(
        '''
            4. MODELLING PREPROCESS

            In this section, we import the necessary machine learning libraries, 
            define the X and y variables for prediction, 
            and split the dataset into training and testing subsets to prepare for model development.

            Importing libraries and divide dataset to train and data.
        '''
    )

    with st.echo():

        from sklearn.model_selection import train_test_split, KFold, cross_val_score
        from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score, roc_curve, roc_auc_score


        from sklearn.model_selection import GridSearchCV

        # Algorithmns models to be compared
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC

    # Creating the X and y variables
    X = df1.drop(columns='Recurred').values
    y = df1["Recurred"].values

    # Spliting X and y into train and test version
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

    st.markdown(
        '''
            4.1 Preparing models to decide which choose:
            - In each iteration, one fold is used as the validation set, and the remaining folds are used for training.
            - This ensures that every data point is used for both training and validation, reducing the risk of overfitting or bias from a single train-test split.
            - Performs cross-validation by running the model through each KFold split.
            - Trains the model on the training portion of each fold and evaluates it on the validation portion using the specified metric (scoring='recall' in this case).
            - The use of RECALL as the evaluation metric to focus on minimizing false negatives.
        '''
    )

    
    # To feed the random state
    seed = 7

    # Prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('SVM', SVC(gamma='auto')))

    # Evaluate each model in turn
    results = []
    names = []
    scoring = 'recall'

    for name, model in models:
        kfold = KFold(
            n_splits=10, 
            shuffle=True, 
            random_state=seed
            )
        cv_results = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=kfold, 
            scoring=scoring
            )
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    df1 = pd.DataFrame(
        data=list(map(list, zip(*results))),
        columns=names
    )

    fig10 = px.box(
        df1,
        color_discrete_sequence=[PLOT_COLORS[1]]
    )

    st.plotly_chart(fig10)

    st.markdown(
        '''
            In the box plot above, the Random Forest (RF) and Support Vector Machine (SVM) models shows best results on cross-validation scores compared to other models. 
            This consistency in performance indicates that the both models are less affected by variations in the data during cross-validation, suggesting a reliable and stable models.
        '''
    )

    ##----------------------- START svm

    # Defining parameter range 
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000], 
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
        'kernel': ['rbf']
        }

    model1 = SVC()
    grid = GridSearchCV(
        model1, 
        param_grid = param_grid, 
        refit = True, 
        verbose = 1
        )

    # fitting the model for grid search 
    grid.fit(X_train, y_train)

    st.markdown(
        f'''
            4.2 Model: Support Vector Machine

            Using GridSearchCV it runs the same loop with cross-validation, to find the best parameter combination. Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to build a single new model using the best parameter setting.
            - Best scorre: {round(grid.best_score_, 4)}
            - Best parameters after tuning: {grid.best_params_}
            - Our model after hyper-parameter tuning: {grid.best_estimator_}
        '''
    )

    ## Define SVM Model with the best parameter setting
    model_svm = SVC(C=1000, gamma=0.0001)

    # Trainning with the best params
    model_svm.fit(X_train, y_train)

    # Predicting using our  model
    y_pred_smv = model_svm.predict(X_test)

    ##----------------------- START rf

    # Defining parameter range 
    param_grid = {
        "max_depth": [3,5, 7, 10,None],
        "n_estimators":[3,5,10,25,50,150],
        "max_features": [4,7,15,20]
        }

    model2 = RandomForestClassifier(random_state=2)
    grid = GridSearchCV(
        model2, 
        param_grid = param_grid, 
        scoring = 'recall', 
        verbose = 1,
        cv = 5
        )

    # fitting the model for grid search 
    grid.fit(X_train, y_train)

    st.markdown(
        f'''
            4.2 Model: Random Forest

            Using GridSearchCV it runs the same loop with cross-validation, to find the best parameter combination. Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to build a single new model using the best parameter setting.
            - Best scorre: {round(grid.best_score_, 4)}
            - Best parameters after tuning: {grid.best_params_}
            - Our model after hyper-parameter tuning: {grid.best_estimator_}
        '''
    )

    ## Define Random Forest Model with the best parameter setting
    model_rf = RandomForestClassifier(max_depth=3, max_features=15, n_estimators=25, random_state=2)

    # Trainning with the best params
    model_rf.fit(X_train, y_train)

    # Predicting using our  model
    y_pred_rf = model_rf.predict(X_test)

    ##-------- Compare models
    col0, col1 = st.columns(2)

    with col0:
        st.markdown(
            f'''
                #### Support Vector Machine
                - Accuracy: {round(accuracy_score(y_test, y_pred_smv), 4)}
                - F-beta score: {round(fbeta_score(y_test, y_pred_smv, beta=2), 4)}
                - Confusion Matrix: {confusion_matrix(y_test, y_pred_smv)}
            '''
        )

    with col1:
        st.markdown(
            f'''
                #### Random Forest
                - Accuracy: {round(accuracy_score(y_test, y_pred_rf), 4)}
                - F-beta score: {round(fbeta_score(y_test, y_pred_rf, beta=2), 4)}
                - Confusion Matrix: {str(confusion_matrix(y_test, y_pred_rf))}
            '''
        )

    # Get predicted probabilities for the positive class
    y_pred_rf_prob = model_rf.predict_proba(X_test)[:, 1]  # Random Forest probabilities
    y_pred_svm_prob = model_svm.decision_function(X_test)  # SVM decision function

    # Compute ROC curve and AUC for Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_prob)
    auc_rf = roc_auc_score(y_test, y_pred_rf_prob)

    # Compute ROC curve and AUC for SVM
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm_prob)
    auc_svm = roc_auc_score(y_test, y_pred_svm_prob)

    # Plot the ROC curves using Plotly
    fig11 = go.Figure()

    # Add ROC curve for Random Forest
    fig11.add_trace(go.Scatter(
        x=fpr_rf, y=tpr_rf,
        mode='lines',
        name=f'Random Forest (AUC = {auc_rf:.2f})',
        line=dict(color='blue')
    ))

    # Add ROC curve for SVM
    fig11.add_trace(go.Scatter(
        x=fpr_svm, y=tpr_svm,
        mode='lines',
        name=f'SVM (AUC = {auc_svm:.2f})',
        line=dict(color='green')
    ))

    # Add diagonal line (random performance)
    fig11.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='black', dash='dash')
    ))

    # Update layout
    fig11.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        legend_title='Models',
        template='plotly_dark'
    )

    st.plotly_chart(fig11)

    y_pred_rf = model_rf.predict([X_test[0]])
    print(y_pred_rf)


    # Write on session state to the other page
    st.session_state['ModelRF'] = model_rf
    st.session_state['dataset'] = df

print('----')