# Yelp-Recommendation-System
Dataset: https://www.yelp.com/dataset
This program used the business.json and review.json. It processes business and review data, trains models to predict restaurant ratings, and visualizes the results. It aims to predict restaurant ratings based on user reviews, providing a recommendation system for plalforms like Yelp.

# Libraries and Constants
## Libraries
- Pandas and NumPy: For data manipulation and analysis.
- Matplotlib and Seaborn: For data visualization.
- Scikit-learn: For model training and evaluation, including KNN and various preprocessing tools.
- Keras: For building and training neural network models.
- Others: Various utilities for data handling and regular expressions.

## Constants
Define file paths, model parameters (like N_FACTORS, BATCH_SIZE, EPOCHS, etc.), and a random state for reproducibility.

# Data Loading and Preprocessing Functions
## load_data(file_path, chunksize=100000)
Loads JSON data in chunks for efficient handling of large datasets.

## preprocess_business_data(business_df)
Cleans the city names and filters states with fewer than 10 businesses.

## get_user_input(business_df)
Interactively asks the user to select a state and city, ensuring valid inputs.

## remove_symbols_and_merge(names)
Cleans restaurant names by removing non-alphanumeric characters and merges similar names.

## extract_keys(attr, key)
Extracts specific keys (like GoodForMeal or Ambience) from a dictionary attribute.

## str_to_dict(attr)
Converts string representations of dictionaries to actual dictionaries.

## preprocess_reviews_data(review_df, state_restaurants)
Selects and merges relevant columns from the review data with the filtered state restaurants data.

# Model Training Functions
## train_knn_model(X_train, y_train)
Uses GridSearchCV to find the best hyperparameters for a K-Nearest Neighbors classifier and trains the model.

## build_keras_model(n_users, n_rests, n_factors, min_rating, max_rating)
Builds a collaborative filtering model using Keras. 
This model includes:
- Embedding layers for users and restaurants.
- Dot product of embeddings.
- Bias terms for users and restaurants.
- Additional dense layers for enhanced learning.
- Output scaled to match the rating range.

## train_keras_model(model, X_train, y_train, X_val, y_val)
Trains the Keras model with early stopping and model checkpointing to save the best model based on validation loss.

## Why does this code trains 2 different models?
It employs both content-based and collaborative filtering techniques to recommend restaurants. 

### KNN model
- Purpose: This model is used for content-based filtering. It recommends restaurants based on the similarity of their attributes (e.g., categories, business parking, ambience).
- Features: Attributes and categories of the restaurants.
- Method: The KNN algorithm finds restaurants that are similar to a given restaurant by comparing the feature vectors of the restaurants. It uses the GridSearchCV to find the best hyperparameters for the KNN model.

### Collaborative Filtering Model using Neural Networks
- Purpose: This model is used for collaborative filtering. It recommends restaurants based on user ratings and interactions.
- Features: User and restaurant interactions (ratings given by users to restaurants).
- Method: The neural network model is built using Keras. It creates embeddings for users and restaurants and learns the latent factors that represent the underlying patterns in user behavior and restaurant characteristics. This model is trained to predict user ratings for restaurants, and these predictions are used to recommend restaurants.

### Why 2 Models?
- Diversity in Recommendations: Using two models allows the recommendation system to capture different aspects of the data. The KNN model focuses on the content and attributes of the restaurants, while the neural network model focuses on the patterns in user interactions.

- Improved Accuracy: Each model can provide recommendations that the other might miss. For example, the KNN model might recommend a restaurant with similar categories, while the neural network model might recommend a restaurant that users with similar tastes have highly rated.

- Complementary Approaches: Content-based filtering is effective when there is ample attribute data but may struggle with new users or items (cold start problem). Collaborative filtering excels in leveraging user behavior data but may struggle with new restaurants or users. Combining both approaches can mitigate the weaknesses of each.

### Workflow in Code
#### Data Preprocessing
- load and preprocess business and review data
- clean and merge data, extract necessary attributes and categories

#### Content-based Filtering
- Train the KNN model on the processed attribute and category data.
- Validate the model and print its accuracy.

#### Collaborative Filtering
- Prepare the data for the neural network model by encoding user and restaurant IDs and creating the user-item interaction matrix.
- Train the neural network model on the user-restaurant rating data.
- Validate the model and plot the distribution of actual vs. predicted ratings.

#### Combining Results
Use embeddings from the neural network model and attributes from the KNN model to find and recommend similar restaurants.

# Evaluation and Visualization
## plot_ratings_distribution(df_test)
Plots histograms comparing the distribution of true and predicted ratings.

# Main Function
## main()
- Loads and preprocesses the business and review data.
- Gets user input for state and city.
- Preprocesses the restaurant names and attributes.
- Encodes user and business IDs as numerical values
- Splits the data into training and testing sets.
- Trains the KNN model and evaluates its accuracy.
- Builds, trains, and evaluates the Keras model.
- Plots the distribution of true and predicted ratings.

# Execution
The script executes the main function if run as a standalone program, initiating the entire recommendation system pipeline.

# Summary
- Data Handling: Efficient loading and preprocessing of large datasets.
- User Interaction: Interactive selection of state and city.
- Model Training: Training both a traditional KNN model and a neural network model for rating predictions.
- Evaluation and Visualization: Evaluating model performance and visualizing rating distributions.
