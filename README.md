# SERP Clustering and Analysis

A Python tool for analyzing Search Engine Results Pages (SERP) data, including keyword clustering, domain analysis, and type distribution analysis.

## Features

### Keyword Clustering
- Identifies related keywords based on shared URLs in SERP results
- Uses graph-based clustering with overlap coefficient
- Identifies main topics for each cluster using TF-IDF
- Generates CSV output with cluster details

### Domain Analysis
- Creates interactive heatmaps of domain rankings
- Analyzes domain overlaps and common keywords
- Generates visualizations for domain distribution

### Type Distribution Analysis
- Analyzes the distribution of SERP element types
- Identifies elements above and below the fold
- Creates interactive visualizations and summary tables

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/serp-clustering.git
cd serp-clustering
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script (`main.py`) serves as a CLI orchestrator to run analyses. It provides a unified interface to run keyword clustering, domain analysis, and type distribution analysis.

### Basic Usage

Run all analyses on your data:
```bash
python src/main.py /path/to/your/data.csv
```

### Command Line Options

The main script supports the following options:

```bash
python src/main.py /path/to/your/data.csv [options]
```

Required arguments:
- `input_file`: Path to your SERP data CSV file

Optional arguments:
- `--analysis {all,serp,domain,type}`: Type of analysis to run (default: all)
- `--keyword_column KEYWORD_COLUMN`: Column name for keywords (default: 'keyword')
- `--url_column URL_COLUMN`: Column name for URLs (default: 'url')
- `--min_common_urls MIN_COMMON_URLS`: Minimum common URLs for clustering (default: 3)
- `--min_overlap MIN_OVERLAP`: Minimum overlap coefficient (default: 0.1)

### Examples

1. Run all analyses with default settings:
```bash
python src/main.py /path/to/your/data.csv
```

2. Run only keyword clustering analysis:
```bash
python src/main.py /path/to/your/data.csv --analysis serp
```

3. Run only domain analysis:
```bash
python src/main.py /path/to/your/data.csv --analysis domain
```

4. Run only type distribution analysis:
```bash
python src/main.py /path/to/your/data.csv --analysis type
```

5. Run analyses with custom column names:
```bash
python src/main.py /path/to/your/data.csv --keyword_column search_term --url_column result_url
```

6. Run with custom clustering parameters:
```bash
python src/main.py /path/to/your/data.csv --min_common_urls 5 --min_overlap 0.2
```

### Advanced Usage

While the main script is the recommended way to run analyses, you can also run individual scripts directly if needed:

1. Keyword Clustering:
```bash
python src/serp_clustering.py /path/to/your/data.csv --keyword_column keyword --url_column url
```

2. Domain Analysis:
```bash
python src/domain_analysis.py /path/to/your/data.csv
```

3. Type Distribution Analysis:
```bash
python src/type_distribution.py --input_file /path/to/your/data.csv
```

## Input Data Format

The input CSV file should contain the following columns:
- `keyword`: The search keyword
- `url`: The URL in the SERP results
- `type`: The type of SERP element
- `rectangle.y`: The vertical position of the element (for above/below fold analysis)

## Output Files

All outputs are saved in the `outputs` directory:

### Keyword Clustering
- `keyword_clusters.csv`: Contains cluster topics, keywords, sizes, and common URLs

### Domain Analysis
- `domain_heatmap.html`: Interactive visualization of domain rankings
- `domain_overlap.html`: Interactive visualization of domain overlaps

### Type Distribution Analysis
- `type_distribution.html`: Interactive visualization of SERP element types
- `type_distribution_table.html`: Summary table of type distribution
- `above_fold_distribution.html`: Interactive visualization of above/below fold distribution
- `above_fold_summary.html`: Summary table of above/below fold analysis

## Dependencies

- pandas
- networkx
- scikit-learn
- plotly
- scipy
- argparse

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 