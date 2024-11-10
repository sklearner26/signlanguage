import streamlit as st
import numpy as np
import pandas as pd
import textdistance
import re
from sign import process_sign_gesture
from collections import Counter
import os

# Function to display the Sign Gesture Prediction page
def show_sign_gesture_page():
    st.markdown('<h1 style="color:red;">Sign Gesture Prediction</h1>', unsafe_allow_html=True)
    st.write('This page will show sign gesture prediction functionality. Currently, this is a placeholder.')
    process_sign_gesture()


def show_sign_data_page():
    st.markdown('<h1 style="color:red;">Sign Gesture Data - A-Z</h1>', unsafe_allow_html=True)
    
    # Create 26 images, one for each letter of the alphabet
    # Assume images are named 'a.jpg', 'b.jpg', ..., 'z.jpg'
    sign_images = [f"sign_images/{chr(65+i)}.jpg" for i in range(26)]  # Replace with your path to images
    
    # Display images in 4 columns per row
    cols = st.columns(4)
    
    # Loop over the images and place them in the columns
    for i in range(0, 26, 4):  # Loop over the images in chunks of 4
        for j, col in enumerate(cols):
            # Calculate the index of the current image
            image_idx = i + j
            if image_idx < 26:  # Ensure we don't go beyond 26 images
                image_path = sign_images[image_idx]
                
                # Check if the image exists at the specified path
                if os.path.exists(image_path):
                    # Display the image and label it with the letter
                    col.image(image_path, caption=chr(65 + image_idx), use_container_width=True)
                else:
                    # If image doesn't exist, display a message
                    col.write(f"Image for '{chr(65 + image_idx)}' not found at path: {image_path}")


def load_and_repeat_words(file_path):
    words = []
    with open(file_path, encoding='utf-8') as f:
        data = f.read()
        data = data.lower()  # Convert to lowercase for uniformity
        word_list = re.findall(r'\w+', data)  # Collect all words
        words += word_list * 50  # Repeat each word 50 times
    return words

# Load and process the first text file
words1 = load_and_repeat_words('/content/words.txt')

# Combine the words and make sure to use only unique words
words = list(set(words1))  # Remove duplicates to get a unique set of words

# Load or create the temp file that will hold additional words for the current session
temp_file = 'temp.txt'

# Function to save the updated word list to the temp file
def save_temp_data(word_list):
    with open(temp_file, 'w', encoding='utf-8') as f:
        for word in word_list:
            f.write(f"{word}\n")

# Function to load the temporary data from the temp file
def load_temp_data():
    if os.path.exists(temp_file):
        with open(temp_file, 'r', encoding='utf-8') as f:
            temp_words = f.read().splitlines()
            return temp_words
    return []

# Function to show the textual data (words)
def show_textual_data_page():
    st.markdown('<h1 style="color:red;">Textual Data - Words from File</h1>', unsafe_allow_html=True)
    
    # Combine loaded words with temp words (session data)
    temp_words = load_temp_data()
    all_words = list(set(words + temp_words))  # Ensure words remain unique by combining both sources

    # Display words in a grid (5 words per row)
    columns = st.columns(5)
    for i, word in enumerate(all_words):
        col = columns[i % 5]
        col.write(word)  # Display word in the respective column

    # Provide an option to add new words
    new_word = st.text_input("Enter a word to add:", "")
    if st.button("Add Word"):
        if new_word:
            # Add the new word to the session (only for the current session)
            new_word_lower = new_word.lower()  # Add lowercase version of the word
            temp_words.append(new_word_lower)
            save_temp_data(temp_words)  # Save to temp.txt file for session persistence
            st.success(f"Word '{new_word}' added!")
        else:
            st.warning("Please enter a word before adding.")

# Function to show the NLP page
def show_nlp_page():
    st.markdown('<h1 style="color:red;">NLP - Next Word Prediction</h1>', unsafe_allow_html=True)
    st.write('Enter a prefix and click "Predict Next Words" to see the most likely next words.')

    # Combine loaded words with temp words (session data)
    temp_words = load_temp_data()
    all_words = list(set(words + temp_words))  # Ensure words remain unique

    # Input field for the user to enter a prefix
    prefix_input = st.text_input('Enter word prefix', '')

    # Button to trigger the prediction
    if st.button('Predict Next Words'):
        if prefix_input:
            # Get the top 5 words for the given prefix from the combined list
            predictions = predict_next_words(prefix_input, all_words, top_n=5)

            # Display the predictions in a table or list format
            if isinstance(predictions, list):
                prediction_df = pd.DataFrame(predictions, columns=['Word', 'Similarity', 'Probability'])
                st.write(prediction_df)
            else:
                st.write(predictions)
        else:
            st.write("Please enter a prefix.")

# Function to predict the next words based on input prefix
def predict_next_words(prefix, word_list, top_n=5):
    """
    Given a word prefix, predict the top_n most likely next words based on frequency and similarity,
    excluding the prefix itself.
    """
    # Count frequency of each word
    word_freq_dict = Counter(word_list)

    # Calculate the relative probability of each word
    N = sum(word_freq_dict.values())
    prob = {w: word_freq_dict[w] / N for w in word_freq_dict}

    # Find words that start with the given prefix, excluding the prefix itself
    matching_words = [word for word in word_freq_dict if word.startswith(prefix) and word != prefix]

    if not matching_words:
        return f"No words found starting with '{prefix}'"

    # Calculate Jaccard similarity for each matching word (similarity can be based on edit distance)
    similarity = []
    for word in matching_words:
        sim = 1 - textdistance.Jaccard(qval=2).distance(prefix, word)  # Using Jaccard distance with qval=2
        similarity.append((word, sim, prob[word]))  # Include word, similarity score, and probability

    # Sort by similarity (descending) and probability (descending), and select the top_n words
    sorted_similarities = sorted(similarity, key=lambda x: (x[1], x[2]), reverse=True)

    # Get the top_n words
    top_predictions = sorted_similarities[:top_n]

    return top_predictions

# Function to show the homepage with title and video
def show_home_page():
    st.markdown(
        """
        <style>
        .title {
            color: red;
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .video-container {
            width: 80%;
            margin: 20px auto;
            display: block;
            text-align: center;
            position: relative;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Title
    st.markdown('<div class="title">Welcome to Indian Sign Language App</div>', unsafe_allow_html=True)

    # Add the video (make sure it is appropriately sized to avoid excess space)
    st.markdown("""
    <div class="video-container">
        <video width="100%" autoplay loop muted>
            <source src="https://cdn.pixabay.com/video/2023/11/11/188742-883619742_tiny.mp4" type="video/mp4"> <!-- Replace with your video URL -->
            Your browser does not support the video tag.
        </video>
    </div>
    """, unsafe_allow_html=True)


# Streamlit layout with sidebar navigation
st.sidebar.title("Indian Sign Language App")
page = st.sidebar.selectbox("Select a Page", ["Home", "NLP - Next Word Prediction", "Textual Data", "Sign Data", "Sign Gesture Prediction"])

# Display corresponding page based on user selection
if page == "Home":
    # Show home page with title and video
    show_home_page()

elif page == "NLP - Next Word Prediction":
    # Show the NLP prediction page
    show_nlp_page()

elif page == "Textual Data":
    # Show textual data page
    show_textual_data_page()

elif page == "Sign Data":
    # Show sign data page (A-Z images)
    show_sign_data_page()

elif page == "Sign Gesture Prediction":
    # Show the sign gesture prediction page
    show_sign_gesture_page()
