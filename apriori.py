import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

precess = pd.read_csv("/Users/unixking/Desktop/머신러닝/recommend/Preprocessed_data.csv")

idx = precess[precess['Category'] == '9'].index
precess.drop(idx , inplace=True)

books = pd.read_csv("/Users/unixking/Desktop/머신러닝/recommend/BX_Books.csv", sep=';', encoding="latin-1", error_bad_lines=False)
users = pd.read_csv("/Users/unixking/Desktop/머신러닝/recommend/BX-Users.csv", sep=';', encoding="latin-1", error_bad_lines=False)
ratings = pd.read_csv("/Users/unixking/Desktop/머신러닝/recommend/BX-Book-Ratings.csv", sep=';', encoding="latin-1", error_bad_lines=False)

precess["Category"] = precess["Category"].str.replace('[', '')
precess["Category"] = precess["Category"].str.replace(']', '')
precess["Category"] = precess["Category"].str.replace('\'', '')
precess["Category"] = precess["Category"].str.replace('\"', '')

ratings

df = precess

df = df[:100000]

df.columns

df.Category


df[df['rating']> 5].shape

df1 = df[df['rating']>5].drop_duplicates()

df1["Category"].nunique()

crosstab = pd.crosstab(df1['user_id'],df['Category']).astype('bool')

freq_item = apriori(crosstab,min_support=0.0001,use_colnames=True)

rules = association_rules(freq_item, metric = "lift", min_threshold = 0.05)

rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])

rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])

def search_apriori(string):
    search_df = rules[rules['antecedents'].str.lower()== string.lower()]
    search_df.sort_values(by='lift', ascending=False)
    search_df = search_df.drop_duplicates()

    return search_df[:10]['consequents'].to_list()

search_apriori('Actresses')

search_apriori('Fiction')

search_apriori("Adolescence")


#--------------------------------------------------------------------------------
mean_users_rating = ratings.groupby('User-ID')['Book-Rating'].mean()
users_rating = ratings.set_index('User-ID')
users_rating['mean-rating'] = mean_users_rating
users_rating.reset_index(inplace = True)
users_rating = users_rating[users_rating['Book-Rating'] > users_rating['mean-rating']]
users_rating['is_fav'] = 1

val = users_rating['User-ID'].value_counts()
list_to_keep = list(val[(val>10) & (val<100)].index)
users_rating = users_rating[users_rating['User-ID'].isin(list_to_keep)]
users_rating.shape


df = pd.pivot_table(users_rating, index = 'User-ID', columns = 'ISBN', values = 'is_fav')
df.fillna(value = 0, inplace = True)
df.shape


from sklearn.decomposition import PCA

pca = PCA(n_components = 3)
pca.fit(df)
pca_fit = pca.transform(df)

pca_fit = pd.DataFrame(pca_fit, index = df.index)
pca_fit

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score ,silhouette_samples

from IPython.display import Image,display
from IPython.core.display import HTML
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

km = KMeans(n_clusters = 3)
plt.rcParams['figure.figsize'] = (16, 9)
clusters =km.fit_predict(pca_fit)
cmhot = plt.get_cmap('brg')
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pca_fit[0], pca_fit[2], pca_fit[1], c = clusters, cmap = cmhot)
plt.title('Data points n 3D PCA axis')
plt.show()


tss = []
for i in range(2, 26):
    km = KMeans(n_clusters = i, random_state = 0)
    km.fit(pca_fit)
    tss.append(km.inertia_)
plt.plot(range(2, 26), tss, '-')
plt.show()



for n in [3,4,5,6,7,8]:
    ax1 = plt.figure().gca()
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(pca_fit) + (n + 1) * 10])
    km = KMeans(n_clusters=n,random_state=0)
    clusters = km.fit_predict(pca_fit)
    silhouette_avg = silhouette_score(pca_fit, clusters)
    print("For n_clusters =", n,
          "The average silhouette_score is :", silhouette_avg)
    silhouette_values = silhouette_samples(pca_fit, clusters)
    y_start = 10
    for i in range(n): 
        ith_cluster = np.sort(silhouette_values[clusters==i])
        cluster_size = ith_cluster.shape[0]
        y_end = y_start + cluster_size 
        ax1.fill_betweenx(np.arange(y_start, y_end),
                          0, ith_cluster)
        ax1.text(-0.05, y_start + 0.5 * cluster_size, str(i))
        y_start = y_end + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n),
                 fontsize=14, fontweight='bold')
plt.show()


kmeans_4 = KMeans(n_clusters = 4, random_state = 0).fit(pca_fit)
df['cluster'] = kmeans_4.labels_
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pca_fit[0], pca_fit[2], pca_fit[1], c = df['cluster'], cmap = cmhot)
plt.title('Data points')
plt.show()
