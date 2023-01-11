from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import streamlit as st
import time

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
    # Load dataframe
    data = pd.read_csv('Characteristics.csv')
    return data

def view_data(data):
    st.dataframe(data)

def view_data_dimensions(data):
    st.write('Data Dimensions:', data.shape)

def view_data_statistics(data):
    st.dataframe(data.describe().round(2))
    results = data.describe(include=['object']).round(2).to_dict()
    st.write(results)

def select_columns(data):
    # Ask user to select columns to query on
    selected_columns = st.multiselect('Select columns to query on:', data.columns)
    return selected_columns

def plot_data(data, column):

    # Check if column is numerical or categorical
    if data[column].dtype in [np.int64, np.float64]:
        # Numerical column
        st.subheader("Numerical column")
        st.info("Plotting histogram...")
        hist_fig = px.histogram(data, x=column)
        st.plotly_chart(hist_fig)
        st.info("Plotting box plot...")
        box_fig = px.box(data, y=column)
        st.plotly_chart(box_fig)
    else:
        # Categorical column
        st.subheader("Categorical column")
        st.info("Plotting bar plot...")
        bar_fig = sns.histplot(data[column])
        fig = bar_fig.get_figure()
        bar_fig = px.bar(data, x=column, y=column, orientation='v')
        st.pyplot(fig)

@st.cache
def convert_df(df):
    return df.to_csv()

def build_and_evaluate_model(data, selected_columns, target_col):

    # Check if target column is numerical or categorical
    if data[target_col].dtype in [np.int64, np.float64]:
        st.subheader("Numerical target column")
        model = RandomForestRegressor(n_jobs=-1)
        metric = mean_absolute_error
        metric_name = 'Mean Absolute Error'
    else:
        st.subheader("Categorical target column")
        model = RandomForestClassifier(n_jobs=-1)
        metric = accuracy_score
        metric_name = 'Accuracy Score'

    ohe_data = pd.get_dummies(data[selected_columns])

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(ohe_data, data[target_col].values, test_size=0.2)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    if data[target_col].dtype in [np.int64, np.float64]:
        score = metric(y_test, y_pred)
    else:
        score = metric(y_test, y_pred)
    st.write(f"{metric_name}: {score:.2f}")
    st.info("Plotting feature importances...")
    feature_importance = pd.Series(model.feature_importances_, index=ohe_data.columns)
    feature_importance.plot.barh(color='green')
    st.pyplot()




def main():

    st.title("Data Management Dashboard")

    # Load data
    st.header("View Data:")
    st.write("Dataset Loaded...")
    data = load_data()

    # View data
    view_data(data)

    st.write("Missing Data:")
    st.write(data.isna().sum().to_dict())

    st.write("Total Duplicates:")
    st.write(data.duplicated().sum())

    if st.button("Remove Duplicates"):
        st.success("Duplicates Removed!")

    if st.button("View Data Dimensions"):
        view_data_dimensions(data)
    
    st.header("Explore Data")

    if st.button("View Data Statistics"):
        with st.spinner("Calculating..."):
            time.sleep(2)
            view_data_statistics(data)


    # Plot data
    column = st.selectbox("Select a column to plot", data.columns)
    if st.button("Plot Data Visualization"):
        plot_data(data, column)
        

    # Select columns to query on
    st.header("Query Data:")
    selected_columns = select_columns(data)
    st.write('You selected:', selected_columns)

    data_len = st.slider("Select Sampling Data Length:", min_value=0, max_value=len(data), value=500)

    # Query data on selected columns
    if selected_columns:
        query_result = data[selected_columns].sample(data_len).reset_index(drop=True)
        st.dataframe(query_result)

        csv = convert_df(query_result)
        st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name='query_data.csv',
                            mime='text/csv',
                        )
                            
    # Build and evaluate model
    st.header("Prediction:")
    target_col = st.selectbox("Select target column", data.columns)
    if st.button("Build and Evaluate Model"):
        with st.spinner("Building Model..."):
            time.sleep(2)
            build_and_evaluate_model(data, selected_columns, target_col)

if __name__ == '__main__':
    main()
