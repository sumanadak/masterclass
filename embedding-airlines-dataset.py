# %%
import pandas as pd
import numpy as np
import altair as alt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# %%
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("api_key")

# %%
import cohere
co = cohere.Client(api_key)

# %%
# Load the dataset to a dataframe
#pd.set_option('display.max_colwidth', None)
df_orig = pd.read_csv('https://raw.githubusercontent.com/cohere-ai/notebooks/main/notebooks/data/atis_intents_train.csv',names=('intent','query'))
#display(df_orig)


# %%
# Take a small sample for illustration purposes
sample_classes = ['atis_airfare', 'atis_airline', 'atis_ground_service']
df = df_orig.sample(frac=0.1, random_state=30)
df = df[df.intent.isin(sample_classes)]
df_orig = df_orig.drop(df.index)
df.reset_index(drop=True,inplace=True)

# Remove unnecessary column 
intents = df['intent'] #save for a later need
df.drop(columns=['intent'], inplace=True)
display(df)
#display(df_orig.loc[[53,2538,4401,3404]])

# %%
def get_embeddings(texts, model="embed-english-v3.0", input_type="search_document"):
    output = co.embed(
        texts=texts, 
        model=model, 
        input_type=input_type, 
        truncate='NONE'
    )
    return output.embeddings

df['query_embeds'] = get_embeddings(df['query'].tolist())
display(df)

# %%
# Function to return the principal components
def get_pca(arr, n):
    pca = PCA(n_components=n)
    embeds_transform = pca.fit_transform(arr)
    return embeds_transform
  
# Reduce embeddings to 2 principal components to aid visualization
embeds = np.array(df['query_embeds'].tolist())
embeds_pc = get_pca(embeds, 10)
embeds_pc

# %%
# Set sample size to visualize
sample = 9

# Reshape the data for visualization purposes
source = pd.DataFrame(embeds_pc)[:sample]
source = pd.concat([source,df['query']], axis=1)
source = source.melt(id_vars=['query'])

# Configure the plot
chart = alt.Chart(source).mark_rect().encode(
    x=alt.X('variable:N', title="Embedding"),
    y=alt.Y('query:N', title='',axis=alt.Axis(labelLimit=500)),
    color=alt.Color('value:Q', title="Value", scale=alt.Scale(
                range=["#917EF3", "#000000"]))
)

result = chart.configure(background='#ffffff'
        ).properties(
        width=700,
        height=400,
        title='Embeddings with 10 dimensions'
       ).configure_axis(
      labelFontSize=15,
      titleFontSize=12)

# Show the plot
result

# %%
# visualizing on a 2D plot

# Function to generate the 2D plot
def generate_chart(df,xcol,ycol,lbl='on',color='basic',title=''):
    chart = alt.Chart(df).mark_circle(size=500).encode(
    x=
    alt.X(xcol,
        scale=alt.Scale(zero=False),
        axis=alt.Axis(labels=False, ticks=False, domain=False)
    ),

    y=
    alt.Y(ycol,
        scale=alt.Scale(zero=False),
        axis=alt.Axis(labels=False, ticks=False, domain=False)
    ),
    
    color= alt.value('#333293') if color == 'basic' else color,
    tooltip=['query']
    )

    if lbl == 'on':
        text = chart.mark_text(align='left', baseline='middle',dx=15, size=13,color='black').encode(text='query', color= alt.value('black'))
    else:
        text = chart.mark_text(align='left', baseline='middle',dx=10).encode()

    result = (chart + text).configure(background="#FDF7F0"
        ).properties(
        width=800,
        height=500,
        title=title
       ).configure_legend(
    orient='bottom', titleFontSize=18,labelFontSize=18)
        
    return result

# %%
sample = 9
# Reduce embeddings to 2 principal components to aid visualization
embeds_pc2 = get_pca(embeds, 2)

# Add the principal components to dataframe
df_pc2 = pd.concat([df, pd.DataFrame(embeds_pc2)], axis=1)

# Plot the 2D embeddings on a chart
df_pc2.columns = df_pc2.columns.astype(str)
generate_chart(df_pc2.iloc[:sample],'0','1',title='2D Embeddings')

# %%
# Define new query
new_query = "How can I find a taxi or a bus when the plane lands?"

# %%
# Get embeddings of the new query
new_query_embeds = get_embeddings([new_query], input_type="search_query")[0]

# %%
# Calculate cosine similarity between the search query and existing queries
def get_similarity(target, candidates):
    # Turn list into array
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target),axis=0)

    # Calculate cosine similarity
    sim = cosine_similarity(target, candidates)
    sim = np.squeeze(sim).tolist()
    sort_index = np.argsort(sim)[::-1]
    sort_score = [sim[i] for i in sort_index]
    similarity_scores = zip(sort_index,sort_score)

    # Return similarity scores
    return similarity_scores

# Get the similarity between the search query and existing queries
similarity = get_similarity(new_query_embeds, embeds[:sample])

# %%
# View the top 5 articles
print('Query:')
print(new_query,'\n')

print('Most Similar Documents:')
for idx, sim in similarity:
    print(f'Similarity: {sim:.2f};', df.iloc[idx]['query'])

# %%
# Create new dataframe and append new query
df_sem = df.copy()
df_sem.loc[len(df_sem.index)] = [new_query, new_query_embeds]

# Reduce embeddings dimension to 2
embeds_sem = np.array(df_sem['query_embeds'].tolist())
embeds_sem_pc2 = get_pca(embeds_sem, 2)

# Add the principal components to dataframe
df_sem_pc2 = pd.concat([df_sem, pd.DataFrame(embeds_sem_pc2)], axis=1)

# %%
# Create column for representing chart legend
df_sem_pc2['Source'] = 'Existing'
df_sem_pc2.at[len(df_sem_pc2)-1, 'Source'] = "New"

# Plot on a chart
df_sem_pc2.columns = df_sem_pc2.columns.astype(str)
selection = list(range(sample)) + [-1]
generate_chart(df_sem_pc2.iloc[selection],'0','1',color='Source',title='Semantic Search')

# %%
# Embed the text for clustering
df['clustering_embeds'] = get_embeddings(df['query'].tolist(), input_type="clustering")
embeds = np.array(df['clustering_embeds'].tolist())

# %%
# Pick the number of clusters
n_clusters = 3

# Cluster the embeddings
kmeans_model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
classes = kmeans_model.fit_predict(embeds).tolist()

# Store the cluster assignments
df_clust = df_pc2.copy()
df_clust['cluster'] = (list(map(str,classes)))

# Preview the cluster assignments
df_clust.head()

# %%
# Plot on a chart
df_clust.columns = df_clust.columns.astype(str)
generate_chart(df_clust.iloc[:sample],'0','1',lbl='on',color='cluster',title='Clustering with 3 Clusters')


