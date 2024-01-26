import pandas as pd
import re
from nltk.stem import PorterStemmer
from collections import Counter

def identify_most_likely_domain(data_file, keyword_file):
    """
    Identifies the most likely domain for the entire DataFrame based on keyword matching.

    Args:
        data_file (str): Path to the DataFrame containing the data to analyze.
        keyword_file (str): Path to the Excel file containing domain-keyword mappings.

    Returns:
        str: The most likely domain for the entire DataFrame.
    """

    stemmer = PorterStemmer()

    # Load data and keywords
    df = pd.read_excel(data_file)
    domains_dict = define_domains_from_excel(keyword_file)

    # Identify potential domains for each row
    def identify_domain_for_row(row):
        text_columns = [col for col in df.columns if df[col].dtype == "object"]
        text_column = next((col for col in text_columns if not pd.api.types.is_numeric_dtype(df[col])), None)
        text = row[text_column] if text_column else ""

        most_related_domain = None
        max_score = 0

        for domain, keywords in domains_dict.items():
            domain_score = 0
            for keyword in keywords:
                matches = [word for word in text.lower().split() if stemmer.stem(keyword).lower() in stemmer.stem(word)]
                domain_score += len(matches)

            if domain_score > max_score:
                max_score = domain_score
                most_related_domain = domain

        return most_related_domain

    predicted_domains = df.apply(identify_domain_for_row, axis=1)

    # Determine the most likely domain overall using majority vote
    domain_counts = Counter(predicted_domains)
    most_likely_domain = domain_counts.most_common(1)[0][0]

    return most_likely_domain

def define_domains_from_excel(excel_file_path):
    """
    Reads domain and keyword mappings from an Excel file.

    Args:
        excel_file_path (str): Path to the Excel file containing the mappings.

    Returns:
        dict: A dictionary mapping domain names to lists of keywords.
    """

    df = pd.read_excel(excel_file_path)
    domains_dict = {}

    for index, row in df.iterrows():
        domain = row["Category"]
        keywords = row["Keywords"].split(",")
        domains_dict.setdefault(domain, []).extend(keywords)

    return domains_dict

# Example usage
most_likely_domain = identify_most_likely_domain("Train_Data\hospital.xlsx", "Keyword.xlsx")
print(f"Most likely domain overall: {most_likely_domain}")

