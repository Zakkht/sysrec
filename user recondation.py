import numpy as np
import pandas as pd
import pymongo
import sns
from scipy.stats import kendalltau, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Sample user-item ratings data
# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["travel_app"]


def sim_usrers(user):
    ##############################################
    rating = db["user rating"]
    cursor_rating = rating.find()
    data_rating = pd.DataFrame(list(cursor_rating))
    visit_with = data_rating.loc[data_rating['username'] == user, "username2"]
    print(visit_with)
    #################################################
    collection = db["users"]

    # Query MongoDB and convert result to pandas DataFrame
    cursor = collection.find()
    data = pd.DataFrame(list(cursor))

    df = pd.DataFrame(data)


    # Create a pivot table of users and their ratings for each place
    pivot_table =  df.pivot_table(index='username', columns='place', values='rating', aggfunc='mean').fillna(0)

    ## Fit a kNN model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(pivot_table)

    # Get the index of the target user (e.g., User1)
    target_user_index = pivot_table.index.get_loc(user)

    # Get the k nearest neighbors for the target user
    _, indices = knn.kneighbors(pivot_table.iloc[target_user_index].values.reshape(1, -1), n_neighbors=len(pivot_table))

    # Get the indices of the nearest neighbors
    neighbor_indices = indices.flatten()

    # Filter out the target user
    recommendations = pivot_table.iloc[neighbor_indices].index.tolist()
    #print(recommendations)

    visit_with_set = set(visit_with)

    filtered_recommendations = [user for user in recommendations if user == user and user not in visit_with_set]
    filtered_recommendations2 = [user for user in recommendations if user == user and user in visit_with_set]

    #print(filtered_recommendations)
   # print(filtered_recommendations2)
    filtered_recommendations.extend(filtered_recommendations2)
    print(filtered_recommendations)



def collab_recommend_user(user):
    collection = db["user rating"]

    # Query MongoDB and convert result to pandas DataFrame
    cursor = collection.find()
    data = pd.DataFrame(list(cursor))

    df = pd.DataFrame(data)
    # Create a matrix with mean ratings
    matrix = df.pivot_table(index='username', columns='username2', values='rating', aggfunc='mean').fillna(0)
    # Calculate cosine similarity between 'user' and all other users

    active_user = matrix.loc[user].values.reshape(1, -1)
    cosine_similarity_matrix = cosine_similarity(matrix, active_user)

    # Use the cosine similarity values to weigh the ratings
    weighted_ratings = matrix.multiply(cosine_similarity_matrix, axis=0)

    # Generate recommendations by summing up the weighted ratings
    recommendations = weighted_ratings.sum(axis=0)

    # Filter out places that the user has already rated
    user_rated = (weighted_ratings.loc[user] != 0)  # # to see accurity ,i should change it to ==0 && change it in spearman's ... the first prob to !=
    recommendations[user_rated] = 0

    ############### the first method (video) ###############
    # print(recommendations)# show me first city recommended
    # print(len(weighted_ratings.iloc[0,:])) #show me the first city
    # print(len(weighted_ratings.iloc[:,0])) # show me first user
    for j in range(len(weighted_ratings.columns)):  # iterate over cities
        sum_weght_ness = 0
        for i in range(len(weighted_ratings.index)):  # iterate over users
            if weighted_ratings.iloc[i, j] != 0:
                sum_weght_ness += cosine_similarity_matrix[i]
        if sum_weght_ness != 0:
            recommendations[j] /= sum_weght_ness
        else:
            recommendations[j] = 0

    ###############
    ############# secind method (gpt) ###########
    #recommendations = recommendations / sum(cosine_similarity_matrix)
    ###############

    # Set 'user's weighted ratings as 0 to avoid recommending places he has already visit
    recommendations = recommendations.sort_values(ascending=False)




    return recommendations

def evaluate_recommendations_spearmans(user, recommendations, ground_truth):

    # Get the list of recommended places
    recommended_places = recommendations[recommendations != 0].index.tolist()
    print("recomended : ", recommended_places)
    # Get the list of actual preferences of the user
    actual_places = ground_truth.loc[user][ground_truth.loc[user] != 0].sort_values(ascending=False).index.tolist()
    print("actual     : ", actual_places)

    # Align the recommended places and actual places lists
    # Ensure both lists have the same length
    min_len = min(len(recommended_places), len(actual_places))
    recommended_places = recommended_places[:min_len]
    actual_places = actual_places[:min_len]

    # Compute the rank of recommended places and actual preferences
    recommended_ranks = [recommended_places.index(place) + 1 for place in actual_places]
    actual_ranks = [actual_places.index(place) + 1 for place in actual_places]

    # Calculate Spearman's rank correlation coefficient
    spearman_score, _ = spearmanr(recommended_ranks, actual_ranks)

    return spearman_score





def evaluate_recommendations_kendall(user, recommendations, ground_truth):

    # Get the list of recommended places
    recommended_places = recommendations[recommendations != 0].index.tolist()
    # print(recommended_places)
    # Get the list of actual preferences of the user
    actual_places = ground_truth.loc[user][ground_truth.loc[user] != 0].sort_values(ascending=False).index.tolist()
    # print(actual_places)

    # Align the recommended places and actual places lists
    # Ensure both lists have the same length
    min_len = min(len(recommended_places), len(actual_places))
    recommended_places = recommended_places[:min_len]
    actual_places = actual_places[:min_len]

    # Compute Kendall's tau coefficient
    kendall_tau, _ = kendalltau(recommended_places, actual_places)

    return kendall_tau



def by_overview(user):
    user_data = db["only_users"]
    # Query MongoDB and convert result to pandas DataFrame
    cursor2 = user_data.find()

    user_data = pd.DataFrame(list(cursor2))
    user_df = pd.DataFrame(user_data)
    user_df['comprt'] = user_df['habits'].astype(str) +  user_df['attitude'].astype(str) +  user_df['behavior'].astype(str) +  user_df['econimic_situation'].astype(str) + user_df['sex'].astype(str)
    data = user_df[['comprt','username']]

    X = np.array(data.comprt)
    text_data = X
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(text_data, show_progress_bar=True)

    X = np.array(embeddings)
    n_comp = 5
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    pca_data = pd.DataFrame(pca.transform(X))
    pca_data.head()
    cos_sim_data = pd.DataFrame(cosine_similarity(X))

    def give_recommendations(index, print_recommendation=False, print_recommendation_plots=False, print_genres=False):
        index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:]
        movies_recomm = data['username'].loc[index_recomm].values
        result = {'Movies': movies_recomm, 'Index': index_recomm}
        if print_recommendation == True:
            print('The watched movie is this one: %s \n' % (data['username'].loc[index]))
            k = 1
            for movie in movies_recomm:
                print('The number %i recommended movie is this one: %s \n' % (k, movie))
        if print_recommendation_plots == True:
            print('The plot of the watched movie is this one:\n %s \n' % (data['comprt'].loc[index]))
            k = 1
            for q in range(len(movies_recomm)):
                plot_q = data['comprt'].loc[index_recomm[q]]
                print('The plot of the number %i recommended movie is this one:\n %s \n' % (k, plot_q))
                k = k + 1
        if print_genres == True:
            print('The genres of the watched movie is this one:\n %s \n' % (data['comprt'].loc[index]))
            k = 1
            for q in range(len(movies_recomm)):
                plot_q = data['comprt'].loc[index_recomm[q]]
                print('The plot of the number %i recommended movie is this one:\n %s \n' % (k, plot_q))
                k = k + 1
        return result

    # Create a dictionary to map usernames to indexes
    username_to_index = {username: index for index, username in enumerate(data['username'])}
    user_index = username_to_index.get(user, None)
    print(user_index)
    print(give_recommendations(user_index,True))






# ############################################################
# ttotal = 0
# collection = db["user rating"]
#
# # Query MongoDB and convert result to pandas DataFrame
# cursor = collection.find()
# df = pd.DataFrame(list(cursor))
# users = df['username'].unique()
# for i in users:
#     print(i)
#     recommendations = collab_recommend_user(i)
#     ground_truth = df.pivot_table(index='username', columns='username2', values='rating', aggfunc='mean').fillna(0)
#
#     spearman_score = evaluate_recommendations_spearmans(i, recommendations, ground_truth)
#     print("Spearman's rank correlation coefficient:", spearman_score)
#     ttotal = ttotal + spearman_score
# print(ttotal/len(users))





#print(collab_recommend_user('zakaria'))
#by_overview('zakaria')
#sim_usrers('zak')