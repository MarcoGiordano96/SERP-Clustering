import plotly.graph_objects as go
import pandas as pd
import argparse
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(description="Analyze domain rankings in SERP data.")
    parser.add_argument('file_path', type=str, help="Path to the input CSV file containing SERP data.")
    parser.add_argument('--num_domains', type=int, default=10,
                        help="Number of top domains to display in the heatmap.")
    parser.add_argument('--output_file', type=str, default='domain_heatmap.html',
                        help="Name of the output HTML file for the heatmap visualization.")
    
    args = parser.parse_args()
    
    # Load data
    df = load_serp_data(args.file_path)
    
    # Create heatmap
    fig = serp_heatmap(df, num_domains=args.num_domains)
    
    # Save visualization
    output_path = Path(args.output_file)
    fig.write_html(str(output_path))
    print(f"Domain ranking heatmap saved to {output_path}")

if __name__ == "__main__":
    main() 