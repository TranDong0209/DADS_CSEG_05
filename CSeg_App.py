import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import base64

# Function to preprocess data
def preprocess_data(df):
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='%Y%m%d')
    df = df.dropna()
    df['Year'] = df['TransactionDate'].dt.year
    df['Month'] = df['TransactionDate'].dt.month
    return df

# Load df_loading DataFrame
df = pd.read_csv('df_loading.csv')

# Load rfm_df  DataFrame
rfm_df = pd.read_csv('df_result.csv')

# Load K-means model
loaded_model = joblib.load('kmeans_model.pkl')

# Preprocess the data
df = preprocess_data(df)

# Calculate monthly transaction statistics
monthly_stats = df.groupby(['Year', 'Month']).agg({
    'CustomerID': 'count',
    'NumCDsPurchased': 'sum',
    'TransactionValue': 'sum'
}).reset_index()

# Define custom labels for metrics
metric_labels = {
    'NumCDsPurchased': 'Number of CDs Purchased',
    'TransactionValue': 'Total Transaction Value',
    'CustomerID': 'Number of Customers'
}
# Function to generate a download link for a DataFrame as a CSV file
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_data.csv">Download Predicted Data (CSV)</a>'
    return href

# Function to plot radar chart
def plot_radar_chart(df, customer_id):
    # Create a copy of the DataFrame for scaling and visualization
    df_scaled = df.copy()

    # Function to scale a column to a 0-100 range with reversed values
    def scale_to_0_100_reversed(column):
        return (column.max() - column) / (column.max() - column.min()) * 100

    # Function to scale a column to a 0-100 range
    def scale_to_0_100(column):
        return (column - column.min()) / (column.max() - column.min()) * 100

    # Scale the numerical columns to a 0-100 range in the copy
    df_scaled['Recency'] = scale_to_0_100_reversed(df_scaled['Recency'])
    df_scaled['Frequency'] = scale_to_0_100(df_scaled['Frequency'])
    df_scaled['Monetary'] = scale_to_0_100(df_scaled['Monetary'])

    # Create a custom radar chart for the specified customer
    customer_data = df_scaled[df_scaled['CustomerID'] == customer_id]

    # Define the radial values and categories
    radial_values = customer_data.iloc[0, 1:].values
    categories = customer_data.columns[1:]

    # Create the custom radar chart
    fig = go.Figure()

    for i in range(len(categories)):
        fig.add_trace(go.Scatterpolar(
            r=[radial_values[i]],
            theta=[categories[i]],
            fill='toself',
            name=categories[i]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                showticklabels=True,
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0%', '25%', '50%', '75%', '100%'],
                range=[0, 100]  # Ensure the radial axis range is from 0 to 100%
            )
        ),
        showlegend=True
    )

    st.write("Customer RFM Characteristic:", customer_id)
    st.plotly_chart(fig)

# Function to calculate and display cluster information
def display_cluster_info(rfm_df):
    # Calculate average values for each RFM_Level and return the size of each segment
    rfm_agg2 = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(0)

    rfm_agg2.columns = rfm_agg2.columns.droplevel()
    rfm_agg2.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
    rfm_agg2['Percent'] = round((rfm_agg2['Count'] / rfm_agg2.Count.sum()) * 100, 2)

    # Reset the index
    rfm_agg2 = rfm_agg2.reset_index()

    # Change the Cluster column datatype into discrete values
    rfm_agg2['Cluster'] = 'Cluster ' + rfm_agg2['Cluster'].astype('str')

    # Print the aggregated dataset
    st.write("Customer Segmentation Information:")
    st.write(rfm_agg2)

    # Create a scatter plot to visualize cluster information
    fig_cluster = px.scatter(
        rfm_agg2,
        x="RecencyMean",
        y="MonetaryMean",
        size="FrequencyMean",
        color="Cluster",
        hover_name="Cluster",
        size_max=100,
        width=1000,
        height=400
    )
    st.plotly_chart(fig_cluster)

# Create a mapping between old cluster names and new cluster names
cluster_mapping = {
    0: 'Inactive Customers',
    1: 'Low-Activity Customers',
    2: 'Engaged Customers',
    3: 'High-Value Customers'
}
def perform_batch_prediction(uploaded_file):
    try:
        # Read the uploaded CSV file into a DataFrame
        input_data = pd.read_csv(uploaded_file)

        # Ensure that the DataFrame has the expected columns (Recency, Frequency, Monetary)
        expected_columns = ['Recency', 'Frequency', 'Monetary']
        if not all(col in input_data.columns for col in expected_columns):
            return 'The uploaded file must contain columns: Recency, Frequency, Monetary'

        # Perform predictions on the input data using the loaded K-means model
        cluster_predictions = loaded_model.predict(input_data)

        # Map cluster numbers to cluster names
        predicted_cluster_names = [cluster_mapping.get(cluster, 'Unknown') for cluster in cluster_predictions]

        # Add the predicted cluster names to the input DataFrame
        input_data['Predicted Cluster'] = predicted_cluster_names

        return input_data

    except Exception as e:
        return f'An error occurred: {str(e)}'

# Create a Streamlit app
#st.title('Customer Segmentation Dashboard')

# Define the navigation options
navigation_options = ['Home', 'Customer Profile', 'Business Dashboard', 'Customer Segmentation']

# Create a sidebar to select the page
page = st.sidebar.selectbox('Select a Page:', navigation_options)

if page == 'Home':
    st.markdown(
        """
        <h1 style='color: blue; font-size: 40px;'>Welcome to the CSEG App </h1>
        <p>This app helps you analyze and understand your customer segments.</p>
        """,
        unsafe_allow_html=True,
    )
    # Provide an example of why customer segmentation is important
    st.subheader('Why Customer Segmentation Matters:')
    st.write("Customer segmentation is crucial for businesses to tailor their marketing strategies, "
             "improve customer experiences, and maximize profitability. For example, by identifying high-value customers, "
             "you can provide them with personalized offers and services to retain their loyalty."           
             )
    st.write("RFM stands for Recency, Frequency, and Monetary value, each corresponding to some key customer trait. These RFM metrics are important indicators of a customer’s behavior because frequency and monetary value affects a customer’s lifetime value, and recency affects retention, a measure of engagement."            
             )
    st.write("RFM is a strategy for analyzing and estimating the value of a customer, based on three data points: Recency (How recently did the customer make a purchase?), Frequency (How often do they purchase), and Monetary Value (How much do they spend?)."            
             )
    st.write("An RFM analysis evaluates which customers are of highest and lowest value to an organization based on purchase recency, frequency, and monetary value, in order to reasonably predict which customers are more likely to make purchases again in the future."            
             )
    st.image('RFM.PNG', use_column_width=True)
    st.image('FRM_metric.PNG', use_column_width=True)

    # Provide links to resources for further learning about customer segmentation
    st.subheader('Learn More:')
    st.markdown("[Customer Segmentation Strategies](https://blog.hubspot.com/service/rfm-analysis)")

# Customer Profile Page
elif page == 'Customer Profile':
    # Header with custom color and font size
    st.markdown("<h1 style='color: blue; font-size: 40px;'>Customer Profiles</h1>", unsafe_allow_html=True)

    # Create a grid layout with 2 columns (2/3 - 1/3)
    col1, col2 = st.columns([2, 1])

    # Left column: Check if the selected_customer_id is in the data and Display the filtered customer profiles
    selected_customer_id = col1.selectbox('Select a Customer ID:', df['CustomerID'].unique())
    filtered_customers = df[df['CustomerID'] == selected_customer_id]

    # Display the filtered customer profiles
    if 'filtered_customers' in locals() and not filtered_customers.empty:
        # Set the subheader font size
        col1.markdown(
            "<h2 style='font-size: 18px;'>Filtered Customer Profiles:</h2>",
            unsafe_allow_html=True
        )
        col1.write(filtered_customers)

        # Calculate and display the total NumCDsPurchased and total TransactionValue
        total_num_cds_purchased = filtered_customers['NumCDsPurchased'].sum()
        total_transaction_value = filtered_customers['TransactionValue'].sum()

        col1.write(f'Total NumCDsPurchased: {total_num_cds_purchased}')
        col1.write(f'Total TransactionValue: {total_transaction_value}')

        # Retrieve and display the Cluster information from rfm_df
        customer_cluster = rfm_df[rfm_df['CustomerID'] == selected_customer_id]['Cluster'].values[0]
        col1.write(f'Customer Segmentation: {customer_cluster}')
        
        # Display the radar chart
        plot_radar_chart(rfm_df, selected_customer_id)
        
    elif search_customer:
        col1.info('No matching customers found.')

    # Right column: Top 10 Customers by Total Transaction Value
    with col2:
        # Set the subheader font size
        st.markdown(
            "<h2 style='font-size: 18px;'>Top 10 Customers by Total Transaction Value:</h2>",
            unsafe_allow_html=True
        )
        top_10_customers = df.groupby('CustomerID')['TransactionValue'].sum().reset_index()
        top_10_customers = top_10_customers.sort_values(by='TransactionValue', ascending=False).head(10)
        st.write(top_10_customers)

# Business Dashboard Page
elif page == 'Business Dashboard':
    # Header with custom color and font size
    st.markdown("<h1 style='color: blue; font-size: 40px;'>Business Activities Dashboard</h1>", unsafe_allow_html=True)

    # Create a grid layout with 2 rows and 2 columns
    col1, col2 = st.columns([3, 1])  # Split the layout into two columns

    # Column 1: Line chart
    with col1:
        # Subheader with custom font size
        st.markdown("<h2 style='font-size: 18px;'>Business Activity</h2>", unsafe_allow_html=True)
        # Allow the user to select a specific year
        selected_year = st.selectbox('Select a Year:', df['Year'].unique())

        if selected_year:
            # Filter the data by the selected year
            filtered_data = monthly_stats[monthly_stats['Year'] == selected_year]

            # Allow the user to choose between NumCDsPurchased, TransactionValue, and Number of Customers
            selected_metric = st.selectbox('Select Metric:', ['NumCDsPurchased', 'TransactionValue', 'CustomerID'])

            if selected_metric:
                # Create an interactive line chart using Plotly Express
                fig = px.line(
                    filtered_data, x='Month', y=selected_metric,
                    title=f'{metric_labels[selected_metric]} by Month in {selected_year}',
                    color_discrete_sequence=['blue'],  # Set line color
                    markers=True,  # Show markers
                    labels={selected_metric: metric_labels[selected_metric]},  # Set custom label
                    width=500,  # Set the width of the line chart
                    height=400,  # Set the height of the line chart
                )
                st.plotly_chart(fig)
        # Call the function to display cluster information
        
        display_cluster_info(rfm_df)

    # Column 2: YTD Total Transaction Value
    with col2:
        # Subheader with custom font size
        st.markdown("<h2 style='font-size: 18px;'>Business Result</h2>", unsafe_allow_html=True)
        # Allow the user to select a specific year
        selected_year_ytd = st.selectbox('Select a Year for YTD Total Transaction Value:', df['Year'].unique())

        if selected_year_ytd:
            # Filter the data by the selected year
            filtered_data_ytd = monthly_stats[monthly_stats['Year'] == selected_year_ytd]

            # Calculate the total transaction value update to date for the current year
            current_month_ytd = filtered_data_ytd['Month'].max()
            total_transaction_value_update_to_date_ytd = filtered_data_ytd[filtered_data_ytd['Month'] <= current_month_ytd]['TransactionValue'].sum()

            # Calculate the percentage of total transaction value achieved compared to the plan
            plan_1997 = 4000000  # 4000k
            plan_1998 = 3000000  # 3000k
            if selected_year_ytd == 1997:
                plan_ytd = plan_1997
            elif selected_year_ytd == 1998:
                plan_ytd = plan_1998
            else:
                plan_ytd = 0
            percentage_achieved_ytd = (total_transaction_value_update_to_date_ytd / plan_ytd) * 100

            # Pie chart to visualize the percentage achieved
            pie_chart_data_ytd = pd.DataFrame({'Category': ['Achieved', 'Remaining'], 'Percentage': [percentage_achieved_ytd, 100 - percentage_achieved_ytd]})
            pie_chart_ytd = px.pie(
                pie_chart_data_ytd, names='Category', values='Percentage',
                title=f'Percentage of Plan Achieved for {selected_year_ytd}',
                width=400,  # Set the width of the pie chart
                height=400,  # Set the height of the pie chart
            )
            st.plotly_chart(pie_chart_ytd)

# Prediction Page
if page == 'Customer Segmentation':
    # Header with custom color and font size
    st.markdown("<h1 style='color: blue; font-size: 40px;'>Customer Segmentation Prediction</h1>", unsafe_allow_html=True)

    # Create an input form for prediction
    #st.subheader('Input Customer Data:')
    st.write("Enter customer data for segmentation prediction. You can also upload a CSV file containing customer data.")

    # Input fields for Recency, Frequency, and Monetary
    recency = st.number_input('Recency (in days):')
    frequency = st.number_input('Frequency:')
    monetary = st.number_input('Monetary Value (USD):')

    # Upload CSV file option for batch prediction
    uploaded_file = st.file_uploader('Upload CSV File for Batch Prediction:', type=['csv'])

    # Perform prediction when the user clicks the "Predict" button
    if st.button('Predict'):
        if uploaded_file:
            # Perform batch prediction on the uploaded CSV file
            batch_predictions = perform_batch_prediction(uploaded_file)
            st.write('Batch Prediction Results:')
            st.write(batch_predictions)

            # Add a download link for the predicted data as a CSV file
            st.markdown(get_table_download_link(batch_predictions), unsafe_allow_html=True)
        else:
            # Perform prediction on the input customer data
            input_data = np.array([[recency, frequency, monetary]])
            cluster_prediction = loaded_model.predict(input_data)[0]
            predicted_cluster_name = cluster_mapping.get(cluster_prediction, 'Unknown')

            st.subheader('Prediction Result:')
            st.write(f'The predicted customer segment is: {predicted_cluster_name}')

    # Function to generate a download link for a DataFrame as a CSV file
    def get_table_download_link(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
        href = f'<a href="data:file/csv;base64,{b64}" download="predicted_data.csv">Download Predicted Data (CSV)</a>'
        return href

# About section in the sidebar
st.sidebar.markdown(
    """
    <h3 style='color: blue;'>About CSEG</h3>
    <p>CSEG (Customer Segmentation by RFM using Machine Learning) is an application designed to help businesses gain deeper insights into their customer base. Leveraging RFM analysis and machine learning, CSEG enables the creation of customer segments with similar characteristics to optimize marketing strategies and enhance the customer experience.</p>
    """,
    unsafe_allow_html=True,
)

# Customercarecenter information
st.sidebar.markdown(
    """
    <h3 style='color: blue;'>Customer Care Center</h3>
    <p>If you have any questions or need assistance, please contact our Customer Care Center.</p>
    """,
    unsafe_allow_html=True,
)

# Hotline phone number with phone symbol
st.sidebar.markdown(
    """
    <p>&#9742; +1-800-999-9999</p>
    """,
    unsafe_allow_html=True,
)

# Email address with email symbol
st.sidebar.markdown(
    """
    <p>\u2709️ <a href="mailto:customersupport@cseg.com">customersupport@cseg.com</a></p>
    """,
    unsafe_allow_html=True,
)
# Development Team section in the sidebar with custom font size and color

st.sidebar.markdown(
    """
    <h3 style='color: blue;'>Development Team</h3>
    """,
    unsafe_allow_html=True,
)
# List of team members and their roles
team_members = [
    {'name': 'Dong, Tran', 'role': 'Data Scientist'},
    {'name': 'Viet, Truong Viet', 'role': 'Data Scientist'}
]

# Display team information
for member in team_members:
    st.sidebar.write(f"{member['name']} - {member['role']}")

