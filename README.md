# SMS and Email Spam Classification Application

A production-ready Machine Learning web application that evaluates input text messages and classifies them as either Spam or Ham (Legitimate). The core architecture leverages a Multinomial Naive Bayes model trained alongside a custom TF-IDF (Term Frequency-Inverse Document Frequency) vectorization pipeline.

## Live Deployment
The application is actively hosted and accessible at the following URL:
https://sms-spam-detection-ml-ilzsapuafzndrgh5hej5cc.streamlit.app/

---

## Technical Specifications
* Frontend Interface: Streamlit Framework
* Machine Learning: Scikit-Learn Ecosystem
* Natural Language Processing: NLTK Data Preprocessing Engine
* Infrastructure: Streamlit Community Cloud (Linux Container Environment)

---

## Testing Instructions for Visitors

To evaluate the predictive capabilities of the model, access the live deployment link above and input text matching the operational categories below.

### 1. Evaluating Spam Detection
Copy and paste the following standard pattern into the interface text container to verify the classification system flags promotional and high-risk signals:

"URGENT! Your mobile number has been selected as the winner of a free 500 USD shopping voucher. To claim this prize, visit the link immediately at http://bit.ly/claim-voucher-now. Terms and conditions apply."

### 2. Evaluating Legitimate Text (Ham)
Input standard, low-variance conversational prose to verify the model maintains baseline consistency for non-malicious text:

"Hey, I am working on the project notebook right now. Let me know if you are free to discuss the deployment details over a call later this evening."

---

## Project Structure and Workflow
1. Data Preprocessing: Incoming strings are lowercased, tokenized, filtered to remove punctuation/stopwords, and stemmed using the NLTK PorterStemmer algorithm.
2. Vectorization: Processed tokens are transformed into a numerical sparse matrix via the pre-trained TF-IDF vectorizer configuration.
3. Inference Pipeline: The serialized Multinomial Naive Bayes model loads the array via the Pickle runtime, processes the weights, and outputs the final prediction to the user interface.
