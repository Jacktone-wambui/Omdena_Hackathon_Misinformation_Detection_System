# Semantic Text Matching Application

This project is a semantic text matching application that uses a pre-trained BERT model to find the most relevant statements from a dataset and website content based on user input. It is built using Streamlit for a user-friendly web interface.

## Features

- Input a sentence to match against a dataset.
- Scrape content from a specified website.
- Find and display the most relevant statement from the dataset and the website.
- Show associated speaker information and labels (e.g., Real/Fake).

## Requirements

Ensure you have Python 3.7 or later installed. The following packages are required:

- `streamlit`
- `pandas`
- `requests`
- `beautifulsoup4`
- `scikit-learn`
- `transformers`
- `torch`

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The application requires a dataset in CSV format named `fnn_train.csv`. This file should include the following columns:

- `statement`: Text statements to match.
- `paragraph_based_content`: Content to retrieve based on matching.
- `speaker`: Speaker associated with each statement.
- `label_fnn`: Label indicating the nature of the statement (e.g., Real/Fake).


## How to Run the Application

1. Clone this repository or download the files.
2. Open a terminal and navigate to the directory containing the application.
3. Run the Streamlit application using the following command:

   ```bash
   streamlit run app.py
   ```

5. Open your web browser and navigate to `http://localhost:8501` to access the application.

## How to Use

1. Enter a sentence in the input field.
2. Provide a URL of the website you want to scrape.
3. Click the "Find Matching Statements" button.
4. View the results, including the matching statement from the dataset and the website.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [BERT](https://github.com/google-research/bert) for semantic understanding.
- [Streamlit](https://streamlit.io/) for creating the web interface.
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for web scraping.

## Contact

For any inquiries or issues, please open an issue in this repository or contact me.
