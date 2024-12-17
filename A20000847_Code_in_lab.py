#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

file_path = 'ratings.csv'  
dataset = pd.read_csv(file_path)

dataset['rating'] = dataset['rating'].apply(lambda x: max(1, min(5, round(x))))

print(dataset.head())


# In[2]:



tnu = dataset['userId'].nunique()
tni = dataset['movieId'].nunique()
ratings_per_product = dataset['movieId'].value_counts()
print(f"Total number of users (tnu): {tnu}")
print(f"Total number of items (tni): {tni}")
print("Number of ratings per product:")
print(ratings_per_product)


# In[3]:


import random
user_item_matrix = dataset.pivot_table(index='userId', columns='movieId', values='rating')

missing_ratings = user_item_matrix.isnull().sum(axis=1).sort_values()
candidate_users = missing_ratings.index.tolist()[:3]  
U1, U2, U3 = candidate_users[0], candidate_users[1], candidate_users[2]

def simulate_missing_ratings(user_id, target_missing):
    user_data = user_item_matrix.loc[user_id].copy()
    rated_items = user_data[user_data.notnull()].index.tolist()
    items_to_remove = random.sample(rated_items, len(rated_items) - target_missing)
    user_data[items_to_remove] = None
    return user_data

user_item_matrix.loc[U1] = simulate_missing_ratings(U1, 2)
user_item_matrix.loc[U2] = simulate_missing_ratings(U2, 3)
user_item_matrix.loc[U3] = simulate_missing_ratings(U3, 5)

print(f"Selected Users:")
print(f"User1 (2 missing ratings): {U1}")
print(f"User2 (3 missing ratings): {U2}")
print(f"User3 (5 missing ratings): {U3}")


# In[4]:


missing_percentage = user_item_matrix.isnull().sum() / len(user_item_matrix) * 100

item1 = missing_percentage[missing_percentage >= 4].sort_values().index[0]  # Closest to 4%
item2 = missing_percentage[missing_percentage >= 10].sort_values().index[0]  # Closest to 10%

print(f"Item1 (closest to 4% missing ratings): {item1}")
print(f"Item2 (closest to 10% missing ratings): {item2}")


# In[6]:


active_user = U1
active_user_ratings = user_item_matrix.loc[active_user].dropna()
co_rated_items = active_user_ratings.index


co_rated_users = user_item_matrix[co_rated_items].dropna(how='all').drop(index=active_user)


No_common_users = co_rated_users.dropna(how='all').shape[0]


No_coRated_items = len(co_rated_items)


print(f"Number of users with co-rated items (No_common_users): {No_common_users}")
print(f"Number of co-rated items (No_coRated_items): {No_coRated_items}")


# In[7]:


result_array = np.array([[No_common_users, No_coRated_items]])

result_array = result_array[result_array[:, 0].argsort()[::-1]]

print("2-D Array (No_common_users | No_coRated_items):")
print(result_array)


# In[8]:


import matplotlib.pyplot as plt
ratings_per_item = dataset['movieId'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.plot(ratings_per_item.index, ratings_per_item.values, linestyle='-', color='orange')
plt.title("Quantity of Ratings for Every Item")
plt.xlabel("Item ID (Movie ID)")
plt.ylabel("Number of Ratings")
plt.grid(True)
plt.show()


# In[9]:


def find_threshold_beta(active_user, co_rate_percentage=0.3):
    
    active_user_ratings = user_item_matrix.loc[active_user].dropna()
    co_rated_items = active_user_ratings.index

    
    co_rated_users = user_item_matrix[co_rated_items].dropna(how='all').drop(index=active_user)

    
    co_rated_counts = co_rated_users.notnull().sum(axis=1)

    
    threshold = int(len(co_rated_items) * co_rate_percentage)

    
    qualifying_users = co_rated_counts[co_rated_counts >= threshold]

    
    beta = len(qualifying_users)

    return beta, threshold


beta_U1, threshold_U1 = find_threshold_beta(U1)
beta_U2, threshold_U2 = find_threshold_beta(U2)
beta_U3, threshold_U3 = find_threshold_beta(U3)


print(f"User1 (U1): Threshold β = {beta_U1} (≥ {threshold_U1} co-rated items)")
print(f"User2 (U2): Threshold β = {beta_U2} (≥ {threshold_U2} co-rated items)")
print(f"User3 (U3): Threshold β = {beta_U3} (≥ {threshold_U3} co-rated items)")


# In[10]:


from sklearn.metrics.pairwise import cosine_similarity


def compute_user_similarity(active_user):
    
    active_user_vector = user_item_matrix.loc[active_user].fillna(0).values.reshape(1, -1)

    
    all_users_matrix = user_item_matrix.fillna(0).values

    
    similarity_scores = cosine_similarity(active_user_vector, all_users_matrix).flatten()

   
    similarity_df = pd.DataFrame({
        'userId': user_item_matrix.index,
        'similarity': similarity_scores
    }).set_index('userId').drop(index=active_user)

    
    similarity_df = similarity_df.sort_values(by='similarity', ascending=False)

    return similarity_df


similarity_U1 = compute_user_similarity(U1)
similarity_U2 = compute_user_similarity(U2)
similarity_U3 = compute_user_similarity(U3)


print("Top similar users for User1 (U1):")
print(similarity_U1.head())

print("\nTop similar users for User2 (U2):")
print(similarity_U2.head())

print("\nTop similar users for User3 (U3):")
print(similarity_U3.head())


# In[11]:


def top_20_percent_users(similarity_df):
    top_20_count = int(len(similarity_df) * 0.2)
    return similarity_df.head(top_20_count)

top_20_U1 = top_20_percent_users(similarity_U1)
top_20_U2 = top_20_percent_users(similarity_U2)
top_20_U3 = top_20_percent_users(similarity_U3)

print("Top 20% closest users for User1 (U1):")
print(top_20_U1)

print("\nTop 20% closest users for User2 (U2):")
print(top_20_U2)

print("\nTop 20% closest users for User3 (U3):")
print(top_20_U3)


# In[12]:


def predict_ratings(active_user, top_users):
    
    active_user_ratings = user_item_matrix.loc[active_user]

    
    unseen_items = active_user_ratings[active_user_ratings.isnull()].index

    
    predictions = {}
    for item in unseen_items:
        numerator, denominator = 0, 0
        for user in top_users.index:
            
            rating = user_item_matrix.loc[user, item]
            if not np.isnan(rating):
                numerator += top_users.loc[user, 'similarity'] * rating
                denominator += abs(top_users.loc[user, 'similarity'])

        
        if denominator > 0:
            predictions[item] = numerator / denominator

    
    return pd.Series(predictions).sort_values(ascending=False)

predictions_U1 = predict_ratings(U1, top_20_U1)
predictions_U2 = predict_ratings(U2, top_20_U2)
predictions_U3 = predict_ratings(U3, top_20_U3)

print("Predicted ratings for User1 (U1):")
print(predictions_U1)

print("\nPredicted ratings for User2 (U2):")
print(predictions_U2)

print("\nPredicted ratings for User3 (U3):")
print(predictions_U3)


# In[13]:


def compute_discounted_similarity(top_users, beta):
    
    DF = 1 - (top_users.index.to_series().rank() / beta)
    DF = DF.clip(lower=0)  

    
    DS = top_users['similarity'] * DF.values

    
    result = pd.DataFrame({
        'userId': top_users.index,
        'Similarity': top_users['similarity'],
        'Discount Factor': DF.values,
        'Discounted Similarity': DS
    }).set_index('userId')

    return result


discounted_U1 = compute_discounted_similarity(top_20_U1, beta_U1)
discounted_U2 = compute_discounted_similarity(top_20_U2, beta_U2)
discounted_U3 = compute_discounted_similarity(top_20_U3, beta_U3)


print("Discount Factor and Discounted Similarity for User1 (U1):")
print(discounted_U1)

print("\nDiscount Factor and Discounted Similarity for User2 (U2):")
print(discounted_U2)

print("\nDiscount Factor and Discounted Similarity for User3 (U3):")
print(discounted_U3)


# In[14]:


def top_20_percent_discounted_users(discounted_df):
    
    top_20_count = int(len(discounted_df) * 0.2)
    
    top_20_users = discounted_df.sort_values(by='Discounted Similarity', ascending=False).head(top_20_count)
    return top_20_users


top_20_discounted_U1 = top_20_percent_discounted_users(discounted_U1)
top_20_discounted_U2 = top_20_percent_discounted_users(discounted_U2)
top_20_discounted_U3 = top_20_percent_discounted_users(discounted_U3)


print("Top 20% closest users for User1 (U1) based on Discounted Similarity:")
print(top_20_discounted_U1)

print("\nTop 20% closest users for User2 (U2) based on Discounted Similarity:")
print(top_20_discounted_U2)

print("\nTop 20% closest users for User3 (U3) based on Discounted Similarity:")
print(top_20_discounted_U3)


# In[15]:


def predict_ratings_with_discounted_similarity(active_user, top_discounted_users):
   
    active_user_ratings = user_item_matrix.loc[active_user]

    
    unseen_items = active_user_ratings[active_user_ratings.isnull()].index

    
    predictions = {}
    for item in unseen_items:
        numerator, denominator = 0, 0
        for user in top_discounted_users.index:
            
            rating = user_item_matrix.loc[user, item]
            if not np.isnan(rating):
                numerator += top_discounted_users.loc[user, 'Discounted Similarity'] * rating
                denominator += abs(top_discounted_users.loc[user, 'Discounted Similarity'])

        
        if denominator > 0:
            predictions[item] = numerator / denominator

    
    return pd.Series(predictions).sort_values(ascending=False)


predictions_U1_DS = predict_ratings_with_discounted_similarity(U1, top_20_discounted_U1)
predictions_U2_DS = predict_ratings_with_discounted_similarity(U2, top_20_discounted_U2)
predictions_U3_DS = predict_ratings_with_discounted_similarity(U3, top_20_discounted_U3)


print("Predicted ratings for User1 (U1) using Discounted Similarity:")
print(predictions_U1_DS)

print("\nPredicted ratings for User2 (U2) using Discounted Similarity:")
print(predictions_U2_DS)

print("\nPredicted ratings for User3 (U3) using Discounted Similarity:")
print(predictions_U3_DS)


# In[ ]:




