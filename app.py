from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
    # Load dataframe
    data = pd.read_csv('Characteristics.csv')
    return data

def view_data(data):
    st.dataframe(data)
    st.write('Data Shape:', data.shape)

def select_columns(data):
    # Ask user to select columns to query on
    selected_columns = st.multiselect('Select columns to query on:', data.columns)
    return selected_columns

def plot_data(data):
    # Ask user for column to plot
    column = st.selectbox("Select a column to plot", data.columns)

    # Check if column is numerical or categorical
    if data[column].dtype in [np.int64, np.float64]:
        # Numerical column
        st.subheader("Numerical column")
        st.write("Plotting histogram...")
        hist_fig = px.histogram(data, x=column)
        st.plotly_chart(hist_fig)
        st.write("Plotting box plot...")
        box_fig = px.box(data, y=column)
        st.plotly_chart(box_fig)
    else:
        # Categorical column
        st.subheader("Categorical column")
        st.write("Plotting bar plot...")
        bar_fig = px.bar(data, x=column, y=column, orientation='v')
        st.plotly_chart(bar_fig)

def build_and_evaluate_model(data, selected_columns, target_col):

    # Check if target column is numerical or categorical
    if data[target_col].dtype in [np.int64, np.float64]:
        st.subheader("Numerical target column")
        model = RandomForestRegressor(n_jobs=-1)
        model_name = 'Random Forest Regressor'
        metric = mean_absolute_error
        metric_name = 'Mean Absolute Error'
    else:
        st.subheader("Categorical target column")
        model = RandomForestClassifier(n_jobs=-1)
        model_name = 'Random Forest Classifier'
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
    st.write("Plotting feature importances...")
    feature_importance = pd.Series(model.feature_importances_, index=ohe_data.columns)
    feature_importance.plot.barh(color='green')
    st.pyplot()




def main():
    # Load data
    data = load_data()

    # View data
    view_data(data)

    # Select columns to query on
    selected_columns = select_columns(data)
    st.write('You selected:', selected_columns)

    # Query data on selected columns
    if selected_columns:
        query_result = data[selected_columns]
        st.dataframe(query_result)

    # Plot data
    plot_data(data)
    
    # Build and evaluate model
    target_col = st.selectbox("Select target column", data.columns)
    if st.button("Build and Evaluate Model"):
        build_and_evaluate_model(data, selected_columns, target_col)

if __name__ == '__main__':
    main()
