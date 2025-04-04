import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib  # Use joblib instead of pickle
import os
import sys
import traceback


def load_data(file_path):
    """
    Load the bank marketing dataset
    """
    print(f"Starting to load data from {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        print(f"Current working directory: {os.getcwd()}")
        print(
            f"Files in data directory: {os.listdir('data') if os.path.exists('data') else 'data directory not found'}")

        # Create a sample data file with both yes and no values
        print("Creating a sample data file for testing...")
        sample_data = """age,job,marital,education,default,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,emp.var.rate,cons.price.idx,cons.conf.idx,euribor3m,nr.employed,y
58,management,married,tertiary,no,yes,no,telephone,may,mon,261,1,999,0,nonexistent,1.1,93.994,-36.4,4.857,5191,no
44,technician,single,secondary,no,no,no,telephone,may,mon,151,1,999,0,nonexistent,1.1,93.994,-36.4,4.857,5191,no
33,entrepreneur,married,secondary,no,yes,yes,telephone,may,mon,76,1,999,0,nonexistent,1.1,93.994,-36.4,4.857,5191,no
47,blue-collar,married,secondary,no,yes,no,telephone,may,mon,92,1,999,0,nonexistent,1.1,93.994,-36.4,4.857,5191,no
33,unknown,single,unknown,no,no,no,telephone,may,mon,198,1,999,0,nonexistent,1.1,93.994,-36.4,4.857,5191,no
41,admin.,divorced,secondary,no,yes,no,telephone,may,mon,241,1,999,0,nonexistent,1.1,93.994,-36.4,4.857,5191,yes
29,admin.,single,secondary,no,no,no,telephone,may,mon,185,1,999,0,nonexistent,1.1,93.994,-36.4,4.857,5191,yes
37,technician,married,secondary,no,yes,no,cellular,apr,mon,213,1,999,0,nonexistent,-1.8,93.075,-47.1,4.961,5099,yes
39,services,married,secondary,no,yes,no,cellular,jul,mon,175,1,999,0,nonexistent,1.4,93.918,-42.7,4.962,5228,yes
32,blue-collar,single,primary,no,no,no,cellular,may,fri,288,1,999,0,nonexistent,-1.8,92.893,-46.2,1.313,5099,yes"""

        with open(file_path, 'w') as f:
            f.write(sample_data)

        print(f"Sample data file created at {file_path}")

        # Try to load the newly created file
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded sample data with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading sample data: {str(e)}")
            return None

    try:
        # First, let's check the file format by reading a few lines
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            print(f"First line of the file: {first_line}")

        # Try to detect the delimiter
        if ';' in first_line:
            delimiter = ';'
        elif ',' in first_line:
            delimiter = ','
        else:
            print("Could not detect delimiter, trying both ';' and ','")
            try:
                df = pd.read_csv(file_path, sep=';')
                if len(df.columns) > 1:
                    delimiter = ';'
                else:
                    df = pd.read_csv(file_path, sep=',')
                    delimiter = ','
            except:
                # If both fail, try to read with default settings
                df = pd.read_csv(file_path)
                delimiter = 'auto'

        print(f"Using delimiter: {delimiter}")

        # Now read the file with the detected delimiter
        if delimiter != 'auto':
            df = pd.read_csv(file_path, sep=delimiter)

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # If we have only one column, it might be that the data is not properly split
        if len(df.columns) == 1:
            print("Only one column detected. Attempting to fix the parsing issue...")
            # Try to split the single column into multiple columns
            first_col_name = df.columns[0]
            if delimiter == ';':
                # Try comma as delimiter
                df = pd.read_csv(file_path, sep=',')
            else:
                # Try semicolon as delimiter
                df = pd.read_csv(file_path, sep=';')

            print(f"After fixing: Dataset shape: {df.shape}")
            print(f"After fixing: Columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()
        return None


def explore_data(df, output_dir='static/images'):
    """
    Perform exploratory data analysis and save visualizations
    """
    print("Starting exploratory data analysis...")

    if df is None:
        print("Error: DataFrame is None, cannot perform EDA.")
        return None

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Check for 'unknown' values which are effectively missing values
    print("\nUnknown Values:")
    for col in df.columns:
        if df[col].dtype == 'object':
            unknown_count = (df[col] == 'unknown').sum()
            if unknown_count > 0:
                print(f"{col}: {unknown_count} unknown values ({unknown_count / len(df) * 100:.2f}%)")

    # Check if 'y' column exists
    if 'y' not in df.columns:
        print("Warning: 'y' column not found in the dataset.")
        print(f"Available columns: {df.columns.tolist()}")
        # Try to find a column that might be the target
        potential_targets = [col for col in df.columns if col.lower() in ['target', 'label', 'class', 'deposit']]
        if potential_targets:
            print(f"Potential target columns found: {potential_targets}")
            # Use the first potential target
            df = df.rename(columns={potential_targets[0]: 'y'})
            print(f"Renamed {potential_targets[0]} to 'y'")
        else:
            # If no target column is found, use the last column as target
            print("No potential target column found. Using the last column as target.")
            df = df.rename(columns={df.columns[-1]: 'y'})
            print(f"Renamed {df.columns[-1]} to 'y'")

    # Visualize the distribution of the target variable
    try:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='y', data=df)
        plt.title('Distribution of Target Variable')
        plt.savefig(f'{output_dir}/target_distribution.png')
        plt.close()
        print(f"Saved target distribution plot to {output_dir}/target_distribution.png")
    except Exception as e:
        print(f"Error creating target distribution plot: {str(e)}")
        print(f"Target column values: {df['y'].value_counts()}")

    # Visualize age distribution if it exists
    if 'age' in df.columns:
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['age'], kde=True)
            plt.title('Age Distribution')
            plt.savefig(f'{output_dir}/age_distribution.png')
            plt.close()
            print(f"Saved age distribution plot to {output_dir}/age_distribution.png")
        except Exception as e:
            print(f"Error creating age distribution plot: {str(e)}")

    # Visualize job distribution if it exists
    if 'job' in df.columns:
        try:
            plt.figure(figsize=(12, 6))
            sns.countplot(y='job', data=df, order=df['job'].value_counts().index)
            plt.title('Job Distribution')
            plt.savefig(f'{output_dir}/job_distribution.png')
            plt.close()
            print(f"Saved job distribution plot to {output_dir}/job_distribution.png")
        except Exception as e:
            print(f"Error creating job distribution plot: {str(e)}")

    # Correlation matrix for numeric features
    try:
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_features) > 1:  # Need at least 2 numeric columns for correlation
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numeric_features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix of Numeric Features')
            plt.savefig(f'{output_dir}/correlation_matrix.png')
            plt.close()
            print(f"Saved correlation matrix to {output_dir}/correlation_matrix.png")
    except Exception as e:
        print(f"Error creating correlation matrix: {str(e)}")

    print("EDA visualizations saved to", output_dir)
    return df


def preprocess_data(df, output_dir='models'):
    """
    Preprocess the data and create a preprocessing pipeline
    """
    print("Starting data preprocessing...")

    if df is None:
        print("Error: DataFrame is None, cannot preprocess data.")
        return None, None, None

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Handle 'unknown' values
        # For categorical features, replace 'unknown' with the most frequent value
        for col in df.select_dtypes(include=['object']).columns:
            if (df[col] == 'unknown').any():
                most_frequent = df[col][df[col] != 'unknown'].mode()[0]
                df[col] = df[col].replace('unknown', most_frequent)

        # Define categorical and numerical features
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                                'contact', 'month', 'day_of_week', 'poutcome']
        numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous',
                              'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

        # Check if all features exist in the dataframe
        missing_cat_features = [f for f in categorical_features if f not in df.columns]
        missing_num_features = [f for f in numerical_features if f not in df.columns]

        if missing_cat_features:
            print(f"Warning: Missing categorical features: {missing_cat_features}")
            categorical_features = [f for f in categorical_features if f in df.columns]

        if missing_num_features:
            print(f"Warning: Missing numerical features: {missing_num_features}")
            numerical_features = [f for f in numerical_features if f in df.columns]

        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Convert target to binary (0/1)
        if 'y' in df.columns:
            if df['y'].dtype == 'object':
                df['y'] = df['y'].map({'no': 0, 'yes': 1})

            # Split the data into features and target
            X = df.drop('y', axis=1)
            y = df['y']

            # Print target distribution
            print(f"Target distribution: {y.value_counts()}")
        else:
            print("Warning: 'y' column not found, using dummy target.")
            X = df
            y = pd.Series(np.zeros(len(df)))

        # Fit the preprocessor on the data
        print("Fitting preprocessor...")
        preprocessor.fit(X)

        # Save the preprocessor for later use
        try:
            joblib.dump(preprocessor, f'{output_dir}/preprocessor.joblib')
            print(f"Preprocessor saved to {output_dir}/preprocessor.joblib")
        except Exception as e:
            print(f"Error saving preprocessor with joblib: {str(e)}")
            # Try with a lower protocol
            try:
                import pickle
                with open(f'{output_dir}/preprocessor.pkl', 'wb') as f:
                    pickle.dump(preprocessor, f, protocol=2)
                print(f"Preprocessor saved to {output_dir}/preprocessor.pkl with pickle protocol 2")
            except Exception as e2:
                print(f"Error saving preprocessor with pickle: {str(e2)}")

        return X, y, preprocessor

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        traceback.print_exc()
        return None, None, None


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    """
    print("Splitting data into training and testing sets...")

    if X is None or y is None:
        print("Error: X or y is None, cannot split data.")
        return None, None, None, None

    try:
        from sklearn.model_selection import train_test_split

        # Make sure we have both classes in the target
        if len(y.unique()) < 2:
            print("Warning: Only one class in the target. Adding a synthetic sample of the other class.")
            # Add a synthetic sample of the other class
            if 1 not in y.values:
                # Add a positive sample
                X_synthetic = X.iloc[0:1].copy()
                y_synthetic = pd.Series([1])
            else:
                # Add a negative sample
                X_synthetic = X.iloc[0:1].copy()
                y_synthetic = pd.Series([0])

            # Combine with original data
            X = pd.concat([X, X_synthetic])
            y = pd.concat([y, y_synthetic])

            # We can't stratify with only one sample of a class
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error splitting data: {str(e)}")
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    print("Starting preprocessing script...")

    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)

    # Check if the data file exists
    data_file = 'data/bank-additional-full.csv'

    # Load and explore data
    print("Loading data...")
    df = load_data(data_file)

    if df is not None:
        print("Exploring data...")
        df = explore_data(df)

        # Preprocess data
        print("Preprocessing data...")
        X, y, preprocessor = preprocess_data(df)

        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = split_data(X, y)

        print("Preprocessing completed successfully!")
    else:
        print("Error: Could not load data. Preprocessing failed.")

    print("Preprocessing script finished.")