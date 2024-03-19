import numpy as np
import pandas as pd
import pymongo
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.stats import kendalltau

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["travel_app"]
collection = db["users"]

# Query MongoDB and convert result to pandas DataFrame
cursor = collection.find()
df = pd.DataFrame(list(cursor))
users = df['username'].unique()


###########################################################
from datetime import date
# i
# I made it to make a recomendation based on the season where am I
# def season_of_date(date):
#     year = date.year
#     seasons = {
#         'spring': (pd.Timestamp(year, 3, 21), pd.Timestamp(year, 6, 20)),
#         'summer': (pd.Timestamp(year, 6, 21), pd.Timestamp(year, 9, 22)),
#         'autumn': (pd.Timestamp(year, 9, 23), pd.Timestamp(year, 12, 20))
#     }
#     for season, (start, end) in seasons.items():
#         if start <= date <= end:
#             return season
#     return 'winter'
#
# today = date.today()
#
#
# # Assuming `df` is your DataFrame with a 'season' column
# df = df[df['season'] == season_of_date(today)]
##############################################
def collab_recommend(user,accurity_score ):
    # Create a matrix with mean ratings
    matrix = df.pivot_table(index='username', columns='place', values='rating', aggfunc='mean').fillna(0)

    # Calculate cosine similarity between 'user' and all other users
    active_user = matrix.loc[user].values.reshape(1, -1)
    cosine_similarity_matrix = cosine_similarity(matrix, active_user)

    # Use the cosine similarity values to weigh the ratings
    weighted_ratings = matrix.multiply(cosine_similarity_matrix, axis=0)

    # Generate recommendations by summing up the weighted ratings
    recommendations = weighted_ratings.sum(axis=0)

    # Filter out places that the user has already rated
    user_rated = (weighted_ratings.loc[user] != 0)  # to see accurity ,i should change it to ==0 && change it in spearman's ... the first prob to !=
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

    if accurity_score == True:
        # Evaluate the recommendations
        ground_truth = df.pivot_table(index='username', columns='place', values='rating', aggfunc='mean').fillna(0)
        kendall_score = evaluate_recommendations_kendall(user, recommendations, ground_truth)
        spearman_score = evaluate_recommendations_spearmans(user, recommendations, ground_truth)

        # print("Spearman's rank correlation coefficient:", spearman_score)
        # print("kendall's rank correlation coefficient:", kendall_score)

        return recommendations, spearman_score, kendall_score
    else:
        return recommendations


def content_base_filtring_geog(user):
    # Filter the DataFrame for the specific user (user)
    user_df = df[(df['username'] == user)]
    other_user = df[(df['username'] != user) & (~df['place'].isin(df[df['username'] == user]['place']))] # in the normal sutiation you should add ~ beside df

    # Expand the lists in the 'geography' column into separate rows
    geography_df = user_df.explode('geography')
    other_geography_df = other_user.explode('geography')

    # Calculate the mean rating for each unique combination of 'place' and 'geography'
    geography_df = geography_df.groupby(['place', 'geography'])['rating'].mean().unstack().fillna(0)
    # Fill the places where the user has visited with 1
    other_geography_df = other_geography_df.groupby(['place', 'geography']).apply(
        lambda x: 1 if not x['rating'].isnull().all() else 0).unstack().fillna(0)
    other_geography_df = other_geography_df.T



    # Normalize data
    divisor = sum(sum(geography_df.values))
    multiplacator = geography_df.shape[0]
    multiplacator = divisor / multiplacator

    normalaized_matrix = geography_df / divisor

    # Calculate the sum of values in each column (each geography)
    sum_row = normalaized_matrix.sum(axis=0)

    # Create a DataFrame from normalaized_matrix['Total' of values of geography in each place]
    final_matrix = pd.DataFrame(sum_row, columns=[user])

    # Ensure both DataFrames align by their index and perform the multiplication
    # Align final_matrix to match the structure of other_geography_df, filling missing rows with 0
    # to have the same shape of place_he_visit matrix
    aligned_final_matrix = final_matrix.reindex(other_geography_df.index, fill_value=0)

    # Perform element-wise multiplication
    result = other_geography_df.multiply(aligned_final_matrix[user], axis=0)

    result = result.sum(axis=0)
    sorted_result = result.sort_values(ascending=False)


    # Convert the result to a pandas Series
    recommended_places = sorted_result[sorted_result != 0]

    return recommended_places



def contente_base_filtring_activ(user):
    # Filter the DataFrame for the specific user (user)
    user_df = df[(df['username'] == user)]
    other_user = df[(df['username'] != user) & (~df['place'].isin(df[df['username'] == user]['place']))] # in the normal sutiation you should add ~ beside df

    # Expand the lists in the 'geography' column into separate rows
    activity_df = user_df.explode('activities')
    other_activity_df = other_user.explode('activities')

    # Calculate the mean rating for each unique combination of 'place' and 'geography'
    activity_df = activity_df.groupby(['place', 'activities'])['rating'].mean().unstack().fillna(0)
    # print(geography_df)
    # Fill the places where the user has visited with 1
    other_activity_df = other_activity_df.groupby(['place', 'activities']).apply(
        lambda x: 1 if not x['rating'].isnull().all() else 0).unstack().fillna(0)
    other_activity_df = other_activity_df.T

    # Normalize data
    divisor = sum(sum(activity_df.values))
    multiplacator = activity_df.shape[0]
    multiplacator = divisor / multiplacator

    normalaized_matrix = activity_df / divisor

    # Calculate the sum of values in each column (each geography)
    sum_row = normalaized_matrix.sum(axis=0)

    # Create a DataFrame from normalaized_matrix['Total' of values of geography in each place]
    final_matrix = pd.DataFrame(sum_row, columns=[user])
    # print(final_matrix)

    # Ensure both DataFrames align by their index and perform the multiplication
    # Align final_matrix to match the structure of other_activity_df, filling missing rows with 0
    # to have the same shape of place_he_visit matrix
    aligned_final_matrix = final_matrix.reindex(other_activity_df.index, fill_value=0)
    # print(aligned_final_matrix)

    # Perform element-wise multiplication
    result = other_activity_df.multiply(aligned_final_matrix[user], axis=0)

    result = result.sum(axis=0)
    sorted_result = result.sort_values(ascending=False)

    # return the predicted rating
    return sorted_result


# Weighted COntent (activitie, geography)
def Content_based(user, cbg_weight=0.4, cba_weight=0.6):
    cfg_recommendations = content_base_filtring_geog(user)

    cba_recommendations = contente_base_filtring_activ(user)

    hybrid_recommendations = (cfg_recommendations * cbg_weight) + (cba_recommendations * cba_weight)

    # Return the content based system
    return hybrid_recommendations.sort_values(ascending=False)


def hybrid_based(user, cbf_weight=0.4, cf_weight=0.6):
    # Calculate collaborative filtering (CF) recommendations for the user
    cf_recommendations = collab_recommend(user,False) # to see accurity i chould add True to enable the score mode and fix it in collab_filtring function

    # Calculate content-based filtering (CBF) recommendations for the user
    cbf_recommendations = Content_based(user)

    # Combine CF and CBF recommendations using weighted averaging
    # The hybrid recommendations are calculated as a weighted sum of CF and CBF recommendations
    hybrid_recommendations = (cf_recommendations * cbf_weight) + (cbf_recommendations * cf_weight)

    # Return the hybrid recommendations
    return hybrid_recommendations.sort_values(ascending=False).fillna(0)


def KNN_recomendation(user):
    # Create a matrix with mean ratings
    matrix = df.pivot_table(index='username', columns='place', values='rating', aggfunc='mean').fillna(0)
    matrix = matrix
    print(matrix)

    ############################# get the index ###############
    index_order = matrix.index.get_indexer([user])
    print(index_order)
    ############################################################

    ################## THE MODEL ###############################
    matrix_df = csr_matrix(matrix.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(matrix_df)

    distance, indice = model_knn.kneighbors(matrix.iloc[index_order, :].values.reshape(1, -1), n_neighbors=len(matrix))
    for i in range(0, len(distance.flatten())):
        if i == 0:
            print("recomendation for {0} : \n".format(matrix.index[index_order]))
        else:
            print(
                "{0} : {1} , with a distence of 2 \n".format(i, matrix.index[indice.flatten()[i]], distance.flatten()))


#############################################################################################################################################################



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


# Example usage
# Assuming user is the username for which recommendations are generated
# and recommendations is the pandas Series containing the recommendations
# and ground_truth is the pandas DataFrame containing the actual preferences
# kendall_tau = evaluate_recommendations_kendall(user, recommendations, ground_truth)


def models(user):
    from sklearn.neighbors import NearestNeighbors

    # Create a pivot table of users and their ratings for each place
    matrix = df.pivot_table(index='username', columns='place', values='rating', aggfunc='mean').fillna(0)
    # Fit a kNN model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(matrix)
    ln = len(matrix)
    # Find k nearest neighbors for the target user
    target_user_ratings = matrix.loc[user].values.reshape(1, -1)  # matrix.drop(user)

    distances, indices = knn.kneighbors(target_user_ratings, n_neighbors=ln)

    # Get the indices of the nearest neighbors
    neighbor_indices = indices.flatten()

    # Calculate the average rating of the nearest neighbors for each place
    neighbor_ratings = matrix.iloc[neighbor_indices]

    average_ratings = neighbor_ratings.mean(axis=0)

    # Filter out places that the user has already rated
    user_rated_places = (matrix.loc[user] != 0)  # ==0 && change it in spearman's

    # print(user_rated_places)
    average_ratings[user_rated_places] = 0

    # Sort the recommendations based on the average ratings
    recommendations = average_ratings.sort_values(ascending=False)

    # Evaluate the recommendations
    ground_truth = df.pivot_table(index='username', columns='place', values='rating', aggfunc='mean').fillna(0)
    # kendall_score = evaluate_recommendations_kendall(user, recommendations, ground_truth)
    # spearman_score = evaluate_recommendations_spearmans(user, recommendations, ground_truth)


    # return recommendations, spearman_score, kendall_score
    return recommendations




def rec_trip_overview(user):
    user_data = db["only_users"]
    trip_data = db["trip_infos"]

    # Query MongoDB and convert result to pandas DataFrame
    cursor = user_data.find()
    cursor2 = trip_data.find()

    user_data = pd.DataFrame(list(cursor))
    trip_data = pd.DataFrame(list(cursor2))
    user_df = pd.DataFrame(user_data)
    active_user = user_df.loc[user_df['username'] == user , 'trip_overview']
    data = trip_data[['overview', 'place']]

    X = np.array(data.overview)

    Y = np.array(active_user)



    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(X, show_progress_bar=True)
    embeddings2 = model.encode(Y, show_progress_bar=True)
    X = np.array(embeddings)
    Y = np.array(embeddings2)
    n_comp = 5
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    cos_sim_data = pd.DataFrame(cosine_similarity(X,Y))
    print(cos_sim_data)

    def give_recommendations( print_recommendation=False, print_recommendation_plots=False, print_genres=False):
        index_recomm = cos_sim_data.sort_values(by=0, ascending=False).index.tolist()
        movies_recomm = data['place'].loc[index_recomm].values
        result = {'Movies': movies_recomm, 'Index': index_recomm} #data['place']
        if print_recommendation == True:
            print('The watched movie is this one: %s \n' % user)
            k = 1
            for movie in movies_recomm:
                print('The number %i recommended movie is this one: %s \n' % (k, movie))
                k+=1
        if print_recommendation_plots == True:
            print('The plot of the watched movie is this one:\n %s \n' % (data['overview']))
            k = 1
            for q in range(len(movies_recomm)):
                plot_q = data['overview'].loc[index_recomm[q]]
                print('The plot of the number %i recommended movie is this one:\n %s \n' % (k, plot_q))
                k = k + 1
        if print_genres == True:
            print('The genres of the watched movie is this one:\n %s \n' % (data['overview']))
            k = 1
            for q in range(len(movies_recomm)):
                plot_q = data['overview'].loc[index_recomm[q]]
                print('The plot of the number %i recommended movie is this one:\n %s \n' % (k, plot_q))
                k = k + 1
        return result

        # Create a dictionary to map usernames to indexes


    print(give_recommendations( True,False,False))



####################################
def score(func):
    score_final_soearman = 0
    score_final_kendall = 0
    for i in users:
        print("This is : ", i, "\n")
        recome, score_spearman, score_kendall = func(i,True)
        score_final_soearman = score_final_soearman + score_spearman
        score_final_kendall = score_final_kendall + score_kendall


    print("-----------------------------------------------------------------------------------------")
    print("-------------------------------THE FINAL ACURITY IS -------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    print("=======  the spearman score final of your model is  : ", score_final_soearman / len(users), "==============")
    print("=======  the kendall  score final of your model is  : ", score_final_kendall / len(users), "===============")
    print("-----------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------")



# Example usage
username = 'hamza'
# print(collab_recommend(username,False))
# print(content_base_filtring_geog(username))
# print(contente_base_filtring_activ(username))
#print(Content_based(username, cbg_weight=0.7, cba_weight=0.3))
#print(hybrid_based(username))
#print(models(username))
rec_trip_overview('zakaria') ## i should add more data with same structer only zakaria and test_user can work here
#print(KNN_recomendation(username))



#score(collab_recommend)
############################################################
# ttotal = 0
# for i in users:
#     print(i)
#     recommendations = Content_based(i)
#     ground_truth = df.pivot_table(index='username', columns='place', values='rating', aggfunc='mean').fillna(0)
#
#     spearman_score = evaluate_recommendations_spearmans(i, recommendations, ground_truth)
#     print("Spearman's rank correlation coefficient:", spearman_score)
#     ttotal = ttotal + spearman_score
# print(ttotal/len(users))