from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import os

# Function to preprocess data and train model
def preprocess_and_train():
    # Load and preprocess the data
    california = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(california.data,
                                                        california.target,
                                                        test_size=0.2,
                                                        random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=[X_train_scaled.shape[1]]),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.1)

    # Save the model with timestamp inside 'model' folder within DAG directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dag_directory = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current DAG file
    model_folder = os.path.join(dag_directory, "model")
    os.makedirs(model_folder, exist_ok=True)  # Create 'model' folder if it doesn't exist
    model.save(os.path.join(model_folder, f"tf-model-1_{timestamp}.keras"))

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 15),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG('California_dataset_on_airflow',
          default_args=default_args,
          description='Train TensorFlow model on California housing dataset',
          schedule_interval='@daily',
          )

# Define tasks
preprocess_and_train_task = PythonOperator(
    task_id='preprocess_and_train',
    python_callable=preprocess_and_train,
    dag=dag,
)

# Set task dependencies
preprocess_and_train_task
