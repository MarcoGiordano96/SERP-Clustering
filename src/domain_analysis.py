import plotly.graph_objects as go
import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from itertools import combinations
import scipy.cluster.hierarchy as sch

def load_serp_data(file_path):
    """
    Load SERP data from CSV file
    """
    df = pd.read_csv(file_path)
    return df

def serp_heatmap(df, num_domains=10, select_domain=None):
    """
    Create a heatmap visualization of domain rankings in SERP data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing SERP data with columns: domain, keyword, rank_group
    num_domains : int, default=10
        Number of top domains to display
    select_domain : str, optional
        Specific domain to analyze
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive heatmap visualization
    """
    df = df.rename(columns={'domain': 'displayLink', 'keyword': 'keyword', 'rank_group': 'rank'})
    top_domains = df['displayLink'].value_counts()[:num_domains].index.tolist()
    top_df = df[df['displayLink'].isin(top_domains) & df['displayLink'].ne('')]

    top_df_counts_means = (top_df
                           .groupby('displayLink', as_index=False)
                           .agg({'rank': ['count', 'mean']}))
    top_df_counts_means.columns = ['displayLink', 'rank_count', 'rank_mean']

    top_df = (pd.merge(top_df, top_df_counts_means)
              .sort_values(['rank_count', 'rank_mean'],
                           ascending=[False, True]))
    rank_counts = (top_df
                   .groupby(['displayLink', 'rank'])
                   .agg({'rank': ['count']})
                   .reset_index())
    rank_counts.columns = ['displayLink', 'rank', 'count']
    summary = (df
               .groupby(['displayLink'], as_index=False)
               .agg({'rank': ['count', 'mean']})
               .sort_values(('rank', 'count'), ascending=False)
               .assign(coverage=lambda df: (df[('rank', 'count')]
                                            .div(df[('rank', 'count')]
                                                 .sum()))))
    summary.columns = ['displayLink', 'count', 'avg_rank', 'coverage']
    summary['displayLink'] = summary['displayLink'].str.replace('www.', '', regex=True)
    summary['avg_rank'] = summary['avg_rank'].round(1)
    summary['coverage'] = (summary['coverage'].mul(100)
                           .round(1).astype(str).add('%'))

    num_queries = df['keyword'].nunique()

    fig = go.Figure()

    fig.add_scatter(x=top_df['displayLink'].str.replace('www.', '', regex=True),
                    y=top_df['rank'], mode='markers',
                    marker={'size': 30, 'opacity': 1/rank_counts['count'].max()})

    fig.add_scatter(x=rank_counts['displayLink'].str.replace('www.', '', regex=True),
                    y=rank_counts['rank'], mode='text',
                    text=rank_counts['count'])

    for domain in rank_counts['displayLink'].unique():
        rank_counts_subset = rank_counts[rank_counts['displayLink'] == domain]
        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[0], mode='text',
                        marker={'size': 50},
                        text=str(rank_counts_subset['count'].sum()))

        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[-1], mode='text',
                        text=format(rank_counts_subset['count'].sum() / num_queries, '.1%'))
        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[-2], mode='text',
                        marker={'size': 50},
                        text=str(round(rank_counts_subset['rank']
                                       .mul(rank_counts_subset['count'])
                                       .sum() / rank_counts_subset['count']
                                       .sum(), 2)))

    minrank, maxrank = int(min(top_df['rank'].unique())), int(max(top_df['rank'].unique()))
    fig.layout.yaxis.tickvals = [-2, -1, 0] + list(range(minrank, maxrank+1))
    fig.layout.yaxis.ticktext = ['Avg. Pos.', 'Coverage', 'Total<br>appearances'] + list(range(minrank, maxrank+1))
    fig.layout.height = max([600, 100 + ((maxrank - minrank) * 50)])
    fig.layout.yaxis.title = 'SERP Rank (number of appearances)'
    fig.layout.showlegend = False
    fig.layout.paper_bgcolor = '#eeeeee'
    fig.layout.plot_bgcolor = '#eeeeee'
    fig.layout.autosize = False
    fig.layout.margin.r = 2
    fig.layout.margin.l = 120
    fig.layout.margin.pad = 0
    fig.layout.hovermode = False
    fig.layout.yaxis.autorange = 'reversed'
    fig.layout.yaxis.zeroline = False
    fig.layout.width = 1100

    return fig

def analyze_domain_overlap(df, top_n_domains=15):
    """
    Analyze how often different domains compete for the same keywords
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing SERP data
    top_n_domains : int, default=15
        Number of top domains to analyze
        
    Returns:
    --------
    tuple
        (overlap_matrix, overlap_percentages, common_keywords)
    """
    # Get top domains by frequency
    top_domains = df['domain'].value_counts().nlargest(top_n_domains).index.tolist()
    
    # Create a dictionary of keywords for each domain
    domain_keywords = {}
    for domain in top_domains:
        domain_keywords[domain] = set(df[df['domain'] == domain]['keyword'])
    
    # Calculate overlap matrix
    overlap_matrix = pd.DataFrame(0, index=top_domains, columns=top_domains)
    overlap_percentages = pd.DataFrame(0.0, index=top_domains, columns=top_domains)
    common_keywords = {}
    
    for dom1, dom2 in combinations(top_domains, 2):
        # Get common keywords
        common = domain_keywords[dom1].intersection(domain_keywords[dom2])
        common_keywords[(dom1, dom2)] = common
        
        # Calculate overlap counts and percentages
        overlap_count = len(common)
        overlap_matrix.loc[dom1, dom2] = overlap_count
        overlap_matrix.loc[dom2, dom1] = overlap_count
        
        # Calculate percentage of keywords that overlap
        pct1 = overlap_count / len(domain_keywords[dom1]) * 100
        pct2 = overlap_count / len(domain_keywords[dom2]) * 100
        overlap_percentages.loc[dom1, dom2] = pct1
        overlap_percentages.loc[dom2, dom1] = pct2
    
    return overlap_matrix, overlap_percentages, common_keywords

def create_overlap_heatmap(overlap_matrix, overlap_percentages):
    """
    Create an interactive heatmap visualization of domain overlaps
    
    Parameters:
    -----------
    overlap_matrix : pandas.DataFrame
        Matrix of overlap counts between domains
    overlap_percentages : pandas.DataFrame
        Matrix of overlap percentages between domains
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive heatmap visualization
    """
    import plotly.graph_objs as go
    import numpy as np
    # Reorder domains using hierarchical clustering for better grouping
    linkage = sch.linkage(overlap_matrix.values, method='average')
    dendro = sch.dendrogram(linkage, no_plot=True)
    idx = dendro['leaves']
    ordered_matrix = overlap_matrix.iloc[idx, :].iloc[:, idx]
    ordered_percentages = overlap_percentages.iloc[idx, :].iloc[:, idx]
    ordered_domains = ordered_matrix.columns

    # Create lower triangle mask (inverted from before)
    mask = np.tril(np.ones_like(ordered_matrix.values, dtype=bool))
    masked_matrix = np.where(mask, ordered_matrix.values, np.nan)
    masked_percentages = np.where(mask, ordered_percentages.values, np.nan)

    # Prepare hover text
    hover_text = np.empty(ordered_matrix.shape, dtype=object)
    for i in range(ordered_matrix.shape[0]):
        for j in range(ordered_matrix.shape[1]):
            if i >= j:  # Only for lower triangle (inverted)
                hover_text[i, j] = (
                    f"<b>{ordered_domains[i]}</b> & <b>{ordered_domains[j]}</b><br>"
                    f"Overlapping keywords: <b>{ordered_matrix.iloc[i, j]}</b><br>"
                    f"Overlap %: <b>{ordered_percentages.iloc[i, j]:.1f}%</b>"
                )
            else:
                hover_text[i, j] = None

    # Value labels (only for overlaps > 10 in lower triangle)
    annotations = []
    for i in range(ordered_matrix.shape[0]):
        for j in range(ordered_matrix.shape[1]):
            if i >= j:  # Only for lower triangle (inverted)
                val = ordered_matrix.iloc[i, j]
                if val > 10:
                    annotations.append(dict(
                        x=ordered_domains[j],
                        y=ordered_domains[i],
                        text=str(val),
                        showarrow=False,
                        font=dict(color='black', size=11, family='Arial'),
                        xanchor='center',
                        yanchor='middle',
                    ))

    # Main title and subtitle
    fig = go.Figure(data=go.Heatmap(
        z=masked_matrix,
        x=ordered_domains,
        y=ordered_domains,
        text=hover_text,
        hoverinfo='text',
        colorscale=[
            [0, 'white'],
            [0.2, '#FFE5CC'],
            [0.4, '#FFCC99'],
            [0.6, '#FFB366'],
            [0.8, '#FF9933'],
            [1, '#FF8000']
        ],
        colorbar=dict(
            title=dict(
                text='Number of<br>Overlapping Keywords',
                font=dict(size=12)
            ),
            thickness=20,
            len=0.8,
            y=0.5
        )
    ))

    # Remove all grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.update_layout(
        title=dict(
            text='<b>Domain Overlap Heatmap (Lower Triangle)</b><br><span style="font-size:14px">Darker colors = more keyword overlap between domains. Value labels shown for overlaps > 10.</span>',
            x=0.5,
            y=0.92,  # Keep title position
            xanchor='center',
            yanchor='top',
            font=dict(size=22)
        ),
        xaxis=dict(
            title='',  # Remove axis title
            tickangle=45,
            side='top',  # Move x-axis labels to top
            tickfont=dict(size=12),
            zeroline=False,
            showline=False  # Remove axis lines
        ),
        yaxis=dict(
            title='',  # Remove axis title
            tickangle=0,
            autorange='reversed',
            tickfont=dict(size=12),
            zeroline=False,
            showline=False  # Remove axis lines
        ),
        width=950,
        height=950,
        margin=dict(l=120, r=40, t=320, b=120),  # Keep large top margin
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13)
    )

    fig.update_traces(showscale=True)
    fig.update_layout(annotations=annotations)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Analyze domain rankings in SERP data.")
    parser.add_argument('file_path', type=str, help="Path to the input CSV file containing SERP data.")
    parser.add_argument('--num_domains', type=int, default=10,
                        help="Number of top domains to display in the heatmap.")
    parser.add_argument('--output_file', type=str, default='domain_heatmap.html',
                        help="Name of the output HTML file for the heatmap visualization.")
    parser.add_argument('--overlap_file', type=str, default='domain_overlap.html',
                        help="Name of the output HTML file for the overlap analysis visualization.")
    
    args = parser.parse_args()
    
    # Load data
    df = load_serp_data(args.file_path)
    
    # Create ranking heatmap
    fig = serp_heatmap(df, num_domains=args.num_domains)
    
    # Create overlap analysis
    overlap_matrix, overlap_percentages, common_keywords = analyze_domain_overlap(df, top_n_domains=args.num_domains)
    overlap_fig = create_overlap_heatmap(overlap_matrix, overlap_percentages)
    
    # Save visualizations
    output_path = Path(args.output_file)
    fig.write_html(str(output_path))
    print(f"Domain ranking heatmap saved to {output_path}")
    
    overlap_path = Path(args.overlap_file)
    overlap_fig.write_html(str(overlap_path))
    print(f"Domain overlap analysis saved to {overlap_path}")
    
    # Print some insights
    print("\nTop domain overlaps:")
    for (dom1, dom2), keywords in common_keywords.items():
        if len(keywords) > 0:
            print(f"\n{dom1} and {dom2}:")
            print(f"Common keywords: {len(keywords)}")
            print(f"Sample keywords: {', '.join(list(keywords)[:5])}")

if __name__ == "__main__":
    main() 