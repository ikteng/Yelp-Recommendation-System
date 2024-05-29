# Import necessary libraries
import numpy as np
import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from keras.layers import Add, Activation, Lambda, Embedding, Reshape, Dot, Input, BatchNormalization, Dense, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
BUSINESS_FILE_PATH = "yelp recommendation system/yelp_academic_dataset_business.json"
REVIEW_FILE_PATH = "yelp recommendation system/yelp_academic_dataset_review.json"
N_FACTORS = 50
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_STATE = 42

# Load JSON file into pandas DataFrame
def load_data(file_path, chunksize=100000):
    return pd.read_json(file_path, lines=True, orient='columns', chunksize=chunksize)

# Preprocess business data
def preprocess_business_data(business_df):
    # Clean city column by stripping whitespaces and capitalizing the first letter of each word
    business_df['city'] = business_df['city'].str.strip().str.title()
    # Get the count of businesses in each state
    state_counts = business_df['state'].value_counts()
    # Filter out states with less than 10 businesses
    valid_states = state_counts[state_counts >= 10].index
    return business_df[business_df['state'].isin(valid_states)]

# Get user input for state and city selection
def get_user_input(business_df):
    valid_state = False
    while not valid_state:
        # Show a list of state abbreviations for user to see
        print("State Abbreviation:\n", business_df['state'].value_counts())
        # Prompt user to input a state abbreviation
        state = input("Enter the state abbreviation (e.g., 'PA'): ").upper()
        # Check if the inputted state abbreviation is in the list and the correct format
        if state in business_df['state'].unique() and len(state) == 2:
            valid_state = True
        else:
            print("Invalid state abbreviation. Please try again.")

    valid_city = False
    while not valid_city:
        # Filter out businesses in selected state and is open
        valid_cities = business_df[(business_df['state'] == state) & (business_df['is_open'] == 1)]['cleaned_city'].value_counts()
        # Filter out cities with less than 10 businesses
        valid_cities = valid_cities[valid_cities > 10]
        # Show user a list of cities in the state
        print("Cities in selected state with more than one restaurant:\n", valid_cities)
        # Prompt user to input the city name
        city_name = input("Enter the city name (e.g., 'Philadelphia'): ").title()
        # Check if user input is in the list
        if city_name in valid_cities.index:
            valid_city = True
        else:
            print("Invalid city name. Please try again.")

    return state, city_name

# Remove symbols from restaurant names and merge similar names
def remove_symbols_and_merge(names):
    cleaned_names = {}
    merged_names = {}

    # Iterate through each name
    for name in names:
        # Remove non-alphanumeric characters
        cleaned_name = re.sub(r'[^\w\s]', '', name).strip()
        if cleaned_name not in merged_names:
            merged_names[cleaned_name] = [name]
        else:
            merged_names[cleaned_name].append(name)
    # Store cleaned names in a dictionary
    for cleaned_name, original_names in merged_names.items():
        for original_name in original_names:
            cleaned_names[original_name] = cleaned_name

    return [cleaned_names[name] for name in names]

# Extract keys from attribute dictionary
def extract_keys(attr, key):
    # If attr and a specific key is in the attr dictionary
    if attr and key in attr:
        # Remove it from the dictionary
        return attr.pop(key)
    # If the key is not present, return an empty dictionary
    return "{}"

# Convert string to dictionary
def str_to_dict(attr):
    try:
        # Try to convert a string representation of a dictionary into an actual dictionary
        return ast.literal_eval(attr) if isinstance(attr, str) else {}
    # If the conversion fails, return an empty dictionary
    except (ValueError, SyntaxError):
        return {}

# Preprocess review data by selecting and merging specific columns from df_review to state_restaurants
def preprocess_reviews_data(review_df, state_restaurants):
    # Select specific columns ('user_id', 'business_id', 'stars', 'date') from the review DataFrame
    df_review = review_df[['user_id', 'business_id', 'stars', 'date']]
    # Merge them with selected columns ('business_id', 'name', 'address', 'price') from the state_restaurants DataFrame based on the common 'business_id' column
    return pd.merge(df_review, state_restaurants[['business_id', 'name', 'address', 'price']], on='business_id')

# Train a k-nearest neighbors classifier using grid search cross-validation
def train_knn_model(X_train, y_train):
    # Grid of parameters which includes the number of neighbors and weight function
    param_grid = {
        'n_neighbors': [10, 20, 30, 40, 50],
        'weights': ['uniform', 'distance']
    }

    # Find the best combination of hyperparameters, searches over the grid of parameters
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    # Print the best parameters
    print(f"Best KNN parameters: {grid_search.best_params_}")
    # Return the best estimator (model)
    return grid_search.best_estimator_

# Build CNN model using Keras for collaborative filtering
def build_keras_model(n_users, n_rests, n_factors, min_rating, max_rating):
    # Functions to embedding layers
    def EmbeddingLayer(n_items, n_factors):
        return lambda x: Reshape((n_factors,))(Embedding(n_items, n_factors, embeddings_initializer='he_normal', embeddings_regularizer=l2(1e-6))(x))

    # Define embedding layers for users
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    ub = EmbeddingLayer(n_users, 1)(user)

    # Define embedding layers for restaurants
    restaurant = Input(shape=(1,))
    m = EmbeddingLayer(n_rests, n_factors)(restaurant)
    mb = EmbeddingLayer(n_rests, 1)(restaurant)   

    # Dot product of user and restaurant embeddings
    x = Dot(axes=1)([u, m])
    # Add user and restaurant biases
    x = Add()([x, ub, mb])
    # Apply batch normalization
    x = BatchNormalization()(x)
    # Sigmoid activation function
    x = Activation('sigmoid')(x)
    # Scale output to match rating range
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

    # Dense layers for additional processing
    x = Concatenate()([u, m, x])
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation='linear')(x)

    # Create and compile the model
    model = Model(inputs=[user, restaurant], outputs=x)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))

    return model

# Train the Keras model
def train_keras_model(model, X_train, y_train, X_val, y_val):
    # Early stopping to stop training if the validation loss doesn't improve for 3 consecutive epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # Save the best model based on the validation loss
    model_checkpoint = ModelCheckpoint('yelp recommendation system\best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    # Train the Keras model for specified batch size and number of epochs
    return model.fit([X_train[:, 0], X_train[:, 1]], y_train, 
                     batch_size=BATCH_SIZE, 
                     epochs=EPOCHS, 
                     verbose=1, 
                     validation_data=([X_val[:, 0], X_val[:, 1]], y_val), 
                     callbacks=[early_stopping, model_checkpoint])

# Plot a histogram comparing the distribution of true and predicted ratings
def plot_ratings_distribution(df_test):
    plt.figure(figsize=(8,6))
    plt.hist(df_test['stars'], bins=5, alpha=0.5, label='True Value')
    plt.hist(df_test['predictions'], bins=5, alpha=0.5, color='orange', label='Predicted Value')
    plt.xlabel('Ratings')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of True and Predicted Ratings')
    plt.show()

# Main function to load data, preprocess it, train models, and visualize results
def main():
    # Load and preprocess data
    business_df = next(load_data(BUSINESS_FILE_PATH))
    business_df = preprocess_business_data(business_df)

    # Get user input for state and city
    state, city_name = get_user_input(business_df)
    state_restaurants = business_df[(business_df['state'] == state) & (business_df['city'] == city_name) & (business_df['is_open'] == 1)].copy()

    # Preprocess restaurant names
    state_restaurants['cleaned_name'] = remove_symbols_and_merge(state_restaurants['name'])
    state_restaurants = state_restaurants.drop_duplicates(subset='cleaned_name', keep='first')

    # Extract attributes
    state_restaurants['attributes'] = state_restaurants['attributes'].apply(str_to_dict)
    state_restaurants['GoodForMeal'] = state_restaurants['attributes'].apply(lambda x: extract_keys(x, 'GoodForMeal'))
    state_restaurants['Ambience'] = state_restaurants['attributes'].apply(lambda x: extract_keys(x, 'Ambience'))

    # Load and preprocess review data
    review_df = next(load_data(REVIEW_FILE_PATH))
    df = preprocess_reviews_data(review_df, state_restaurants)

    # Encode user_id and business_id as numerical values
    user_enc = LabelEncoder()
    df['user'] = user_enc.fit_transform(df['user_id'].values)
    rest_enc = LabelEncoder()
    df['rest'] = rest_enc.fit_transform(df['business_id'].values)
    n_users, n_rests = df['user'].nunique(), df['rest'].nunique()

    # Create user and item matrices
    df = df[['user', 'rest', 'stars']]
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    X_train = train[['user', 'rest']].values
    y_train = train['stars'].values
    X_test = test[['user', 'rest']].values
    y_test = test['stars'].values

    # Train KNN model
    knn_model = train_knn_model(X_train, y_train)

    # Predict using KNN model
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Model Accuracy: {accuracy}")

    # Train Keras model
    keras_model = build_keras_model(n_users, n_rests, N_FACTORS, df['stars'].min(), df['stars'].max())
    history = train_keras_model(keras_model, X_train, y_train, X_test, y_test)

    # Evaluate Keras model
    y_pred = keras_model.predict([X_test[:, 0], X_test[:, 1]]).flatten()
    df_test = pd.DataFrame({'user': X_test[:, 0], 'rest': X_test[:, 1], 'stars': y_test, 'predictions': y_pred})

    # Plot rating distributions
    plot_ratings_distribution(df_test)

# Execute main function
if __name__ == "__main__":
    main()
