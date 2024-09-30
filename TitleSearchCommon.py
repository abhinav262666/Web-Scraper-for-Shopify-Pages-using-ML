import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import re
import json

chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless Chrome (optional)
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model (optional)
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems (optional)



def preprocess_data_from_website(text, stopwords, exclude_words):
    # Step 1: Convert text to lowercase
    text_lower = text.lower().strip()

    # Step 2: Remove text if it ends with a stopword
    words = text_lower.split()
    if words and words[-1] in stopwords:
        return None

    # Step 3: Remove text if it contains excluded words
    if any(word in text_lower for word in exclude_words):
        return None

    # Step 4: Remove text if it only contains special characters
    if re.fullmatch(r'\W+', text_lower):  # Matches only non-alphanumeric characters
        return None

    return text


def get_xpath(unique_df):


    df = unique_df

    # Load the extracted data
    extracted_data = pd.read_csv('extracted_data_solawave.csv')

    # Convert to lowercase for matching
    df['Matched Product Title'] = df['Matched Product Title'].str.lower()
    df['Lower Inner Text'] = df['Lower Inner Text'].str.lower()
    extracted_data['text'] = extracted_data['text'].str.lower()

    # Initialize the xpath column
    df['xpath'] = ''

    for index, row in df.iterrows():
        matched_title = row['Matched Product Title']
        fuzzy_score = row['Fuzzy Score']
        lower_inner_text = row['Lower Inner Text']
        
        # Step 1: First search for h1-h6 and p tags
        matches = extracted_data[(extracted_data['text'] == matched_title) & 
                                (extracted_data['tag'].isin(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']))]
        
        # Step 2: If no match is found, search for 'a' tags
        if matches.empty:
            matches = extracted_data[(extracted_data['text'] == matched_title) & (extracted_data['tag'] == 'a')]
        
        # Step 3: If no match is found and the fuzzy score is 100, retry with Lower Inner Text
        if matches.empty and fuzzy_score >= 75:
            # Step 4: First search for h1-h6 and p tags using Lower Inner Text
            matches = extracted_data[(extracted_data['text'] == lower_inner_text) & 
                                    (extracted_data['tag'].isin(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']))]
            
            # Step 5: If still no match, search for 'a' tags using Lower Inner Text
            if matches.empty:
                matches = extracted_data[(extracted_data['text'] == lower_inner_text) & (extracted_data['tag'] == 'a')]
        
        # Step 6: If there are matches, extract the first matching xpath
        if not matches.empty:
            df.at[index, 'xpath'] = matches.iloc[0]['xpath']
        else:
            # If no match is found, set NaN in xpath
            df.at[index, 'xpath'] = None

# Drop rows where xpath is None
    df_filtered = df.dropna(subset=['xpath'])

    # Reset index
    df_filtered.reset_index(drop=True, inplace=True)

    # Display the resulting dataframe
    return df_filtered




def get_similarity_df(df, product_titles):

    import pandas as pd
    import numpy as np
    from gensim.models import Word2Vec
    from sklearn.metrics.pairwise import cosine_similarity
    from fuzzywuzzy import fuzz
    import re

    # Sample product_titles and inner_text (replace with your actual data)
    # Ensure product_titles is lowercase for consistent matching
    product_titles = [title.lower() for title in product_titles]

    # Prepare the sentences for Word2Vec model (tokenize your product titles and inner texts)
    sentences = [title.split() for title in product_titles] + [text.split() for text in df['inner-text'].str.lower()]

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Function to get average Word2Vec embeddings for a sentence
    def get_sentence_vector(sentence, model):
        vector = np.zeros(model.vector_size)
        count = 0
        for word in sentence.split():
            if word in model.wv:
                vector += model.wv[word]
                count += 1
        if count > 0:
            vector /= count
        return vector

    # Store matched titles and corresponding inner text along with cosine scores
    matched_data = []

    # Compute embeddings for product titles
    product_vectors = np.array([get_sentence_vector(title, word2vec_model) for title in product_titles])

    # Compute embeddings for inner texts (lowercase)
    inner_vectors = np.array([get_sentence_vector(text, word2vec_model) for text in df['inner-text'].str.lower()])

    # Compute cosine similarity between each inner text and all product titles
    for i, inner_vector in enumerate(inner_vectors):
        cosine_similarities = cosine_similarity([inner_vector], product_vectors)
        
        # Get the index of the best match
        best_match_idx = cosine_similarities.argmax()
        best_match_score = cosine_similarities[0, best_match_idx]
        
        # Set a threshold for the minimum acceptable match score (e.g., 0.2 or 20%)
        if best_match_score >= 0.5:
            matched_data.append((df['inner-text'][i], df['inner-text'][i].lower(), product_titles[best_match_idx], best_match_score))

    # Convert the matched data to a DataFrame with original and lower-case inner text, matched product titles, and cosine scores
    matched_df = pd.DataFrame(matched_data, columns=['Original Inner Text', 'Lower Inner Text', 'Matched Product Title', 'Cosine Score'])

    # Preprocessing function to remove brackets and content inside them
    def preprocess_title(title):
        return re.sub(r'\s*\(.*?\)\s*', ' ', title).strip()  # Remove brackets and trim whitespace

    # Apply preprocessing to 'Matched Product Title'
    matched_df['Matched Product Title'] = matched_df['Matched Product Title'].apply(preprocess_title)

    # Now we calculate the fuzzy matching score for each row in `matched_df`
    fuzzy_scores = []

    # Iterate through each row and calculate the fuzzy score
    for index, row in matched_df.iterrows():
        lower_inner_text = row['Lower Inner Text']
        matched_title = row['Matched Product Title']
        
        # Compute the fuzzy score using partial ratio
        fuzzy_score = fuzz.partial_ratio(lower_inner_text, matched_title)
        
        # Append the fuzzy score to the list
        fuzzy_scores.append(fuzzy_score)

    # Add fuzzy scores as a new column to the DataFrame
    matched_df['Fuzzy Score'] = fuzzy_scores

    # Drop duplicates to ensure uniqueness in the DataFrame
    unique_df = matched_df.drop_duplicates(subset=['Original Inner Text', 'Matched Product Title'])

    # Optionally, reset the index for a clean DataFrame
    unique_df.reset_index(drop=True, inplace=True)

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Function to calculate cosine similarity using BOW embeddings
    def cosine_similarity_bow(sentence1, sentence2):
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform([sentence1, sentence2])
        similarity = cosine_similarity(bow_matrix[0:1], bow_matrix[1:2])
        return similarity[0][0]

    # Calculate BOW cosine similarity for each row in `unique_df`
    unique_df['BOW-cosine-similarity'] = unique_df.apply(
        lambda row: cosine_similarity_bow(row['Lower Inner Text'], row['Matched Product Title']),
        axis=1
    )

    unique_df.reset_index(drop=True, inplace=True)

    return unique_df




if __name__ == "__main__":

    PATH = "./chromedriver.exe"
    service = Service(PATH)  # Create a Service object with the path

    # Initialize the Chrome driver using the Service object
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Function to extract HTML using Selenium and then parse with Beautiful Soup
    def extract_html_with_selenium(url):
        # Open the webpage
        driver.get(url)

        # Wait for the page to fully load, including lazy-loaded content
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Optionally, scroll down the page to trigger lazy loading (if applicable)
        SCROLL_PAUSE_TIME = 2  # Time to pause after scrolling
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down to the bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait to load the page
            time.sleep(SCROLL_PAUSE_TIME)
            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Get the full HTML page content after scrolling
        html = driver.page_source

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        return soup

    # Recursive function to find parent <li> or <section> tag
    def find_parent_with_tag(element):
        # Look for parent tags like <li> or <section>
        while element:
            parent = element.find_parent(['li', 'section'])
            if parent:
                return parent
            element = element.parent
        return None

    # Function to find elements with ₹ or $ and extract text from parent tag
    def extract_currency_text(soup):
        data = []
        counter = 1

        # Search for all occurrences of ₹ or $ in the HTML
        for symbol in soup.find_all(string=lambda text: text and ('₹' in text or '$' in text or 'Rs.' in text)):
            # Find the parent <li> or <section> for each symbol
            parent = find_parent_with_tag(symbol)
            
            if parent:
                # Extract all the inner text of the parent, split into separate lines/parts for clarity
                inner_text_parts = parent.get_text(separator="\n", strip=True).split("\n")
                
                # Add each part as a separate entry in the DataFrame
                for part in inner_text_parts:
                    if part.strip():  # Ensure it's not an empty string
                        data.append({"inner-text": part, "counter": counter})
                        counter += 1
        
        return pd.DataFrame(data)

    # Function to extract and render only the content from the <main> tag
    def extract_main_content(soup):
        # Find the <main> tag content
        main_content = soup.find('main')
        
        if main_content:
            return main_content
        else:
            return None

    # Example usage
    # url = "https://patternbeauty.com"  # Replace with the actual URL
    # url = "https://global.solawave.co/collections/shop-all"
    url = "https://www.mykitsch.com/products/castor-oil-shampoo-conditioner-bar-combo-2pc"

    # Extract the fully loaded HTML using Selenium and parse it with Beautiful Soup
    soup = extract_html_with_selenium(url)

    # Extract the content from the <main> tag
    main_content = extract_main_content(soup)

    if main_content:
        # Extract currency-related information from the <main> tag
        df = extract_currency_text(main_content)

    # Close the driver
    driver.quit()
    

    # INTIAL PREPROCESSING

    stopwords = ['the', 'a', 'an', 'for', 'and', 'or', 'to', 'at', 'by', 'in', 'on', 'with', 'from', 'of', 'per']
    exclude_words = ['price', 'add to cart', 'review', 'rating', 'shop', 'off', '$', '₹','sold','sale','title']

    # Apply the preprocessing function to the 'inner-text' column of the dataframe
    df['inner-text'] = df['inner-text'].apply(lambda x: preprocess_data_from_website(x, stopwords, exclude_words))

    # Drop any rows where 'inner-text' is None (after processing)
    df = df.dropna().reset_index(drop=True)

    with open('kitsch_products.json', 'r') as file:
       data = json.load(file)
    
    # product_titles = [product['title'].lower() for product in data['products']]
    product_titles = [product['title'].lower() for product in data]


    unique_df = get_similarity_df(df,product_titles)

    unique_df['avg_score'] = (unique_df['Cosine Score']*100 + unique_df['Fuzzy Score'] + 100*unique_df['BOW-cosine-similarity'])  /3

    df_highest_avg_score = unique_df.loc[unique_df.groupby('Matched Product Title')['avg_score'].idxmax()]
    unique_df = df_highest_avg_score.reset_index(drop=True)
   
    unique_df = unique_df[unique_df['avg_score']>75]
    unique_df.reset_index(drop=True, inplace=True)
    unique_df = get_xpath(unique_df)
    print(unique_df)
    unique_df.to_csv("Kitsch_Search.csv")

    
    




    
    

