import requests
from bs4 import BeautifulSoup
import pandas as pd
from nrclex import NRCLex
import plotly.express as px
import plotly.io as pio
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download()
### nltk.download([
#    "abc", "alpino","averaged_perceptron_tagger","averaged_perceptron_tagger_ru",\
#    "basque_grammars","bcp47","biocreative_ppi", "bllip_wsj_no_aux","brown", "brown_tei",
#    "cess_cat", "cess_esp","chat80", "city_database","cmudict", "comtrans",
#    "conll2000", "conll2002","conll2007", "crubadan","dolch", "europarl_raw",
#    "extended_omw", "floresta","framenet_v15", "framenet_v17","gazetteers", "genesis",
#    "gutenberg","ieer", "inaugural", "indian","jeita", "kimmo","knbc", "large_grammars",
#    "lin_thesaurus", "mac_morpho", "machado", "masc_tagged", "moses_sample", "movie_reviews",
#    "mte_teip5", "mwa_ppdb", "names", "nombank.1.0", "nps_chat", "omw-1.4", "omw", "opinion_lexicon",
#    "panlex_swadesh", "paradigms", "pe08", "perluniprops",
#    "pil","pl196x", "porter_test", "ppattach","problem_reports", "product_reviews_1",\
#    "product_reviews_2","propbank","pros_cons","ptb","punkt", "qc","reuters", "rslp","rte","semcor",
#    "senseval","sentiwordnet","shakespeare", "sinica_treebank","smultron","snowball_data",
#    "spanish_grammars", "state_union", "stopwords", "subjectivity", "swadesh", "switchboard",
#    "tagsets","timit","toolbox","treebank","udhr2","udhr","unicode_samples","vader_lexicon","verbnet3",
#    "verbnet","webtext","wmt15_eval","word2vec_sample","wordnet2021","wordnet2022",
#    "wordnet31","wordnet","wordnet_ic","words","ycoe",
# ])

# Task 1
# Web scraping and analysis
# This Jupyter notebook includes some code to get you started with web scraping.
# We will use a package called `BeautifulSoup` to collect the data from the web.
# Once you've collected your data and saved it into a local `.csv` file you should start with your analysis.
# Scraping data from Skytrax
# If you visit [https://www.airlinequality.com] you can see that there is a lot of data there.
# For this task, we are only interested in reviews related to British Airways and the Airline itself.
# If you navigate to this link: [https://www.airlinequality.com/airline-reviews/british-airways] you will see this data.
# Now, we can use `Python` and `BeautifulSoup`to collect all the links to the reviews
# and then to collect the text data on each of the individual review links.

# %%
base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

reviews = []
nu_state = {}
idea_state = {'Total_idea': 1000, 'strongly_positive': 0, 'middle_positive': 0,
              'middle_negative': 0, 'strongly_negative': 0}


def remove_stop_words(input):
    sw_nltk = stopwords.words('english')
    n_reviews = []
    input = input.split(" ")
    for w_1 in input:
        if w_1 in sw_nltk:
            continue
        else:
            n_reviews.append(w_1)
    return str(n_reviews)


def senti_state(input):
    state_1 = SentimentIntensityAnalyzer()
    return state_1.polarity_scores(input)


# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    # print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    nu_id_state = 1
    for idea in reviews:
        nu_state[nu_id_state] = senti_state(idea)
        nu_id_state += 1

for k_1 in nu_state:
    v_1 = nu_state[k_1]
    for k_2 in v_1:
        v_2 = v_1[k_2]
        if k_2 == 'compound':
            if v_2 > 0.50:
                idea_state['strongly_positive'] += 1
            elif 0.0 < v_2 < 0.50:
                idea_state['middle_positive'] += 1
            elif -0.50 < v_2 < 0.0:
                idea_state['middle_negative'] += 1
            elif v_2 > -0.50:
                idea_state['strongly_negative'] += 1

emotion_df = pd.DataFrame.from_dict(idea_state, orient='index')
emotion_df = emotion_df.reset_index()
emotion_df = emotion_df.rename(columns={'index': 'Emotion Classification', 0: 'Emotion Count'})
emotion_df = emotion_df.sort_values(by=['Emotion Count'], ascending=False)
fig_y = px.bar(emotion_df, x='Emotion Count', y='Emotion Classification', color='Emotion Classification', orientation='h', width=800, height=400)
# fig.show()
pio.write_image(fig_y, "ver.png")

df = pd.DataFrame()
df["reviews"] = reviews
df.head()
df.to_csv("BA_reviews.csv")
