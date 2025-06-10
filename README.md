# SERP Clustering and Analysis

This project provides tools for analyzing Search Engine Results Page (SERP) data, including keyword clustering, domain analysis, and type distribution analysis. It's particularly useful for SEO analysis and understanding search result patterns.

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
- Analyzes domain overlap and competition
- Provides detailed metrics for top-performing domains

### 3. Type Distribution Analysis
- Analyzes the distribution of different result types in SERPs
- Creates interactive visualizations of type frequencies
- Provides above/below fold analysis
- Generates detailed summary tables

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
- `--output_file`: Name of the output CSV file (default: 'outputs/keyword_clusters.csv')

### Domain Analysis

```bash
python src/domain_analysis.py path/to/your/serp_data.csv
```

The script will generate:
- Domain ranking heatmap
- Domain overlap analysis
- Detailed domain competition metrics

### Type Distribution Analysis

```bash
python src/type_distribution.py --input_file path/to/your/serp_data.csv
```

The script will generate:
- Type distribution visualization
- Above/below fold analysis
- Detailed summary tables

## Input Data Format

The scripts expect a CSV file with the following columns:
- `keyword`: The search query/keyword
- `rank_group`: The position in search results
- `domain`: The domain name
- `url`: The full URL
- `title`: The page title
- `description`: The meta description
- `type`: The type of search result
- `rectangle.y`: The vertical position of the result

## Output

All outputs are saved in the `outputs` directory:

### Keyword Clustering
- `keyword_clusters.csv`: Contains:
  - Cluster topic
  - Keywords in the cluster
  - Cluster size
  - Common URLs
  - URL count

### Domain Analysis
- `domain_heatmap.html`: Interactive heatmap showing:
  - Domain rankings distribution
  - Coverage percentage
  - Average position
  - Total appearances
- `domain_overlap.html`: Interactive visualization of domain competition

### Type Distribution Analysis
- `type_distribution.html`: Interactive visualization of result types
- `type_distribution_table.html`: Detailed type distribution summary
- `above_fold_distribution.html`: Above/below fold analysis
- `above_fold_summary.html`: Fold distribution summary

## Dependencies

- pandas >= 2.0.0
- networkx >= 3.0
- scikit-learn >= 1.7.0
- plotly >= 6.1.0
- numpy >= 1.22.0
- scipy >= 1.10.0
- argparse >= 1.4.0

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 