# SERP Clustering and Domain Analysis

This project provides tools for analyzing Search Engine Results Page (SERP) data, including keyword clustering and domain ranking analysis. It's particularly useful for SEO analysis and understanding search result patterns.

## Features

### 1. Keyword Clustering
- Groups related keywords based on shared URLs in search results
- Uses graph-based clustering with overlap coefficient
- Identifies main topics for each cluster using TF-IDF analysis
- Outputs detailed cluster information including size, keywords, and common URLs

### 2. Domain Analysis
- Creates interactive heatmaps of domain rankings
- Shows domain coverage and average positions
- Visualizes ranking distribution across different positions
- Provides detailed metrics for top-performing domains

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd serp_clustering_project
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Keyword Clustering

```bash
python src/main.py path/to/your/serp_data.csv --keyword_column keyword --url_column url
```

Optional parameters:
- `--min_common_urls`: Minimum number of common URLs to consider keywords related (default: 3)
- `--min_overlap`: Minimum overlap coefficient (default: 0.1)
- `--output_file`: Name of the output CSV file (default: 'keyword_clusters.csv')

### Domain Analysis

```bash
python src/domain_analysis.py path/to/your/serp_data.csv --num_domains 15
```

Optional parameters:
- `--num_domains`: Number of top domains to display (default: 10)
- `--output_file`: Name of the output HTML file (default: 'domain_heatmap.html')

## Input Data Format

The scripts expect a CSV file with the following columns:
- `keyword`: The search query/keyword
- `rank_group`: The position in search results
- `domain`: The domain name
- `url`: The full URL
- `title`: The page title
- `description`: The meta description

## Output

### Keyword Clustering
- CSV file containing:
  - Cluster topic
  - Keywords in the cluster
  - Cluster size
  - Common URLs
  - URL count

### Domain Analysis
- Interactive HTML heatmap showing:
  - Domain rankings distribution
  - Coverage percentage
  - Average position
  - Total appearances

## Dependencies

- pandas >= 2.0.0
- networkx >= 3.0
- scikit-learn >= 1.7.0
- plotly >= 6.1.0
- numpy >= 1.22.0

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 