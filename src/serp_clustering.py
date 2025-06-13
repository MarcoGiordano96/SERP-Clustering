import pandas as pd
from collections import defaultdict
from itertools import combinations
from urllib.parse import urlparse
import networkx as nx
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

def load_serp_data(file_path):
    """
    Load DataForSEO SERP data from a CSV file
    """
    df = pd.read_csv(file_path)
    return df

def extract_keyword_urls(df, keyword_col, url_col):
    """
    Extract keyword and URL data from the DataFrame using specified column names
    """
    keyword_urls = defaultdict(set)
    
    # Group by keyword and collect URLs for each keyword
    for _, row in df.iterrows():
        keyword = row[keyword_col]
        url = row[url_col]
        
        if pd.notna(keyword) and pd.notna(url):
            # Store the full URL
            keyword_urls[keyword].add(url)
    
    return keyword_urls

def calculate_overlap_coefficient(set1, set2):
    """
    Calculate the overlap coefficient between two sets
    Overlap coefficient = |A ∩ B| / min(|A|, |B|)
    """
    if not set1 or not set2:
        return 0
        
    intersection = len(set1.intersection(set2))
    return intersection / min(len(set1), len(set2))

def identify_cluster_topic(keywords, keyword_urls):
    """
    Identify the main topic of a cluster by finding the most representative keyword
    """
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Get all keywords in the cluster
    keyword_list = list(keywords)
    
    # Create TF-IDF matrix
    try:
        tfidf_matrix = vectorizer.fit_transform(keyword_list)
        
        # Calculate average similarity of each keyword to others
        avg_similarities = []
        for i in range(len(keyword_list)):
            # Get similarity scores with all other keywords
            similarities = []
            for j in range(len(keyword_list)):
                if i != j:
                    # Calculate cosine similarity
                    similarity = (tfidf_matrix[i] * tfidf_matrix[j].T).toarray()[0][0]
                    similarities.append(similarity)
            
            # Calculate average similarity
            avg_similarity = np.mean(similarities) if similarities else 0
            avg_similarities.append(avg_similarity)
        
        # Find the keyword with highest average similarity
        best_idx = np.argmax(avg_similarities)
        return keyword_list[best_idx]
    except:
        # If TF-IDF fails (e.g., all keywords are too short), return the shortest keyword
        return min(keyword_list, key=len)

def cluster_related_keywords(keyword_urls, min_common_urls=3, min_overlap=0.1):
    """
    Cluster keywords that have similar URL patterns using graph-based clustering
    """
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (keywords)
    for keyword in keyword_urls.keys():
        G.add_node(keyword)
    
    # Add edges between related keywords
    for kw1, kw2 in combinations(keyword_urls.keys(), 2):
        urls1 = keyword_urls[kw1]
        urls2 = keyword_urls[kw2]
        
        common_urls = urls1.intersection(urls2)
        if len(common_urls) >= min_common_urls:
            overlap = calculate_overlap_coefficient(urls1, urls2)
            if overlap >= min_overlap:
                G.add_edge(kw1, kw2, weight=overlap)
    
    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))
    
    # Format clusters with additional information
    cluster_results = []
    for i, cluster in enumerate(clusters):
        cluster_list = list(cluster)
        # Get all URLs in the cluster
        cluster_urls = set()
        for keyword in cluster_list:
            cluster_urls.update(keyword_urls[keyword])
        
        # Identify the main topic of the cluster
        main_topic = identify_cluster_topic(cluster_list, keyword_urls)
            
        cluster_results.append({
            'cluster_topic': main_topic,
            'keywords': cluster_list,
            'size': len(cluster_list),
            'common_urls': list(cluster_urls),
            'url_count': len(cluster_urls)
        })
    
    return sorted(cluster_results, key=lambda x: x['size'], reverse=True)

def main():
    parser = argparse.ArgumentParser(description="Cluster keywords based on shared URLs in SERP data.")
    parser.add_argument('file_path', type=str, help="Path to the input CSV file containing SERP data.")
    parser.add_argument('--min_common_urls', type=int, default=3,
                        help="Minimum number of common URLs required to consider two keywords related.")
    parser.add_argument('--min_overlap', type=float, default=0.1,
                        help="Minimum overlap coefficient required to consider two keywords related.")
    parser.add_argument('--output_file', type=str, default='outputs/keyword_clusters.csv',
                        help="Name of the output CSV file for keyword clusters.")
    parser.add_argument('--keyword_column', type=str, default='keyword',
                        help="Name of the column containing keywords in the input CSV.")
    parser.add_argument('--url_column', type=str, default='url',
                        help="Name of the column containing URLs in the input CSV.")

    args = parser.parse_args()

    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # File path to your CSV data
    file_path = args.file_path
    
    # Load and process data
    df = load_serp_data(file_path)
    keyword_urls = extract_keyword_urls(df, args.keyword_column, args.url_column)
    
    # Find keyword clusters
    clusters = cluster_related_keywords(keyword_urls, min_common_urls=args.min_common_urls, min_overlap=args.min_overlap)
    
    if clusters:
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(clusters)
        print(f"Found {len(clusters)} keyword clusters")
        print("\nTop clusters by size:")
        print(results_df[['cluster_topic', 'size', 'keywords']].head(10))  # Show top 10 clusters
        
        # Save results to CSV
        output_file_name = args.output_file
        results_df.to_csv(output_file_name, index=False)
        print(f"\nResults saved to {output_file_name}")
    else:
        print("No keyword clusters found with the specified criteria")

if __name__ == "__main__":
    main() 