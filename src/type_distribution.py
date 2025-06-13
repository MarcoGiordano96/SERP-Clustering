import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import argparse
import os

def analyze_above_fold(df):
    # Define above fold threshold (in pixels)
    above_fold_threshold = 400

    # Create a new column indicating whether the element is above the fold
    df['is_above_fold'] = df['rectangle.y'] < above_fold_threshold

    # Count how many elements are above/below the fold
    above_fold_count = df['is_above_fold'].sum()
    below_fold_count = len(df) - above_fold_count

    # Create a horizontal bar chart for above/below fold distribution
    fig = go.Figure(data=[go.Bar(
        y=['Above Fold', 'Below Fold'],
        x=[above_fold_count, below_fold_count],
        text=[above_fold_count, below_fold_count],
        textposition='outside',
        textfont=dict(
            family='Trebuchet MS, sans-serif',
            size=14,
            color='rgba(50, 50, 50, 0.8)'
        ),
        orientation='h',
        marker=dict(
            color=['#FF8000', '#FFB366'],
            line=dict(color='rgba(0,0,0,0.3)', width=1.5)
        ),
        hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Percentage: %{customdata:.2f}%<extra></extra>',
        customdata=[(above_fold_count/len(df)*100), (below_fold_count/len(df)*100)]
    )])

    # Update layout
    fig.update_layout(
        title={
            'text': 'Distribution of Elements Above/Below Fold',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family='Raleway, sans-serif',
                size=24,
                color='rgba(0,0,0,0.85)'
            )
        },
        yaxis_title="",
        xaxis_title="",
        template="plotly_white",
        height=400,
        width=800,
        yaxis={
            'tickfont': dict(
                family='Roboto, sans-serif',
                size=16,
                color='rgba(0,0,0,0.75)'
            ),
            'gridwidth': 0.1,
            'gridcolor': 'rgba(0,0,0,0.05)'
        },
        xaxis={
            'showgrid': True,
            'gridwidth': 0.1,
            'gridcolor': 'rgba(0,0,0,0.05)',
            'zeroline': False,
            'tickfont': dict(
                family='Roboto, sans-serif',
                size=14,
                color='rgba(0,0,0,0.6)'
            )
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Montserrat, sans-serif",
            bordercolor="rgba(0,0,0,0.2)"
        ),
        margin=dict(l=20, r=60, t=80, b=40),
        bargap=0.25,
        plot_bgcolor='rgba(250,250,250,0.95)'
    )

    # Save the visualization
    fig.write_html('outputs/above_fold_distribution.html')

    # Create a summary table
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Position', 'Count', 'Percentage (%)'],
            fill_color='rgba(82, 45, 128, 0.8)',
            align='center',
            font=dict(
                family='Raleway, sans-serif',
                color='white', 
                size=15
            )
        ),
        cells=dict(
            values=[
                ['Above Fold', 'Below Fold'],
                [above_fold_count, below_fold_count],
                [f"{(above_fold_count/len(df)*100):.2f}%", f"{(below_fold_count/len(df)*100):.2f}%"]
            ],
            fill_color=[['rgba(237, 231, 246, 0.5)', 'rgba(248, 245, 255, 0.5)']],
            align='center',
            font=dict(
                family='Roboto, sans-serif',
                size=14,
                color=['rgba(0,0,0,0.7)']
            ),
            height=30
        ))
    ])

    fig_table.update_layout(
        title={
            'text': "Above/Below Fold Summary",
            'font': dict(
                family='Raleway, sans-serif',
                size=22,
                color='rgba(0,0,0,0.85)'
            )
        },
        height=200,
        margin=dict(l=10, r=10, t=50, b=15)
    )

    fig_table.write_html('outputs/above_fold_summary.html')

    # Print summary statistics
    print("\nAbove/Below Fold Analysis:")
    print(f"Total elements: {len(df)}")
    print(f"Elements above the fold: {above_fold_count} ({above_fold_count/len(df)*100:.2f}%)")
    print(f"Elements below the fold: {below_fold_count} ({below_fold_count/len(df)*100:.2f}%)")

    return above_fold_count, below_fold_count

def analyze_type_distribution(df):
    # Check if 'type' column exists
    if 'type' not in df.columns:
        print("Column 'type' not found. Available columns are:", df.columns.tolist())
        return
    
    # Count the frequency of each type
    type_counts = df['type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    
    # Calculate percentages
    type_counts['Percentage'] = (type_counts['Count'] / type_counts['Count'].sum() * 100).round(2)
    
    # Create a horizontal bar chart with Plotly
    fig = go.Figure()
    
    # Color scale for gradient effect
    colors = px.colors.sequential.Plasma
    
    # Add the horizontal bar chart with custom styling
    fig.add_trace(go.Bar(
        y=type_counts['Type'],
        x=type_counts['Count'],
        text=type_counts['Count'],
        textposition='outside',
        textfont=dict(
            family='Trebuchet MS, sans-serif',
            size=14,
            color='rgba(50, 50, 50, 0.8)'
        ),
        orientation='h',
        marker=dict(
            color=type_counts['Count'],
            colorscale=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1.5)
        ),
        hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Percentage: %{customdata:.2f}%<extra></extra>',
        customdata=type_counts['Percentage']
    ))
    
    # Customize layout
    fig.update_layout(
        title={
            'text': 'Distribution of Result Types',
            'y':0.97,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family='Raleway, sans-serif',
                size=28,
                color='rgba(0,0,0,0.85)'
            )
        },
        yaxis_title="",
        xaxis_title="",
        template="plotly_white",
        height=max(500, len(type_counts) * 45),  # Dynamic height based on number of categories
        width=950,
        yaxis={
            'categoryorder':'total ascending',
            'tickfont': dict(
                family='Roboto, sans-serif',
                size=16,
                color='rgba(0,0,0,0.75)'
            ),
            'gridwidth': 0.1,
            'gridcolor': 'rgba(0,0,0,0.05)'
        },
        xaxis={
            'showgrid': True,
            'gridwidth': 0.1,
            'gridcolor': 'rgba(0,0,0,0.05)',
            'zeroline': False,
            'tickfont': dict(
                family='Roboto, sans-serif',
                size=14,
                color='rgba(0,0,0,0.6)'
            )
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Montserrat, sans-serif",
            bordercolor="rgba(0,0,0,0.2)"
        ),
        margin=dict(l=20, r=60, t=80, b=40),
        bargap=0.25,  # Gap between bars
        plot_bgcolor='rgba(250,250,250,0.95)'  # Slight off-white background
    )
    
    # Add subtle grid lines
    fig.update_yaxes(showgrid=False)
    
    # Save the interactive chart
    fig.write_html('outputs/type_distribution.html')
    
    # Create and save the distribution summary table
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Type', 'Count', 'Percentage (%)'],
            fill_color='rgba(82, 45, 128, 0.8)',  # Deep purple header
            align='center',
            font=dict(
                family='Raleway, sans-serif',
                color='white', 
                size=15
            )
        ),
        cells=dict(
            values=[type_counts['Type'], type_counts['Count'], type_counts['Percentage']],
            fill_color=[['rgba(237, 231, 246, 0.5)', 'rgba(248, 245, 255, 0.5)'] * (len(type_counts)//2 + 1)],
            align='center',
            font=dict(
                family='Roboto, sans-serif',
                size=14,
                color=['rgba(0,0,0,0.7)']
            ),
            height=30
        ))
    ])
    
    fig_table.update_layout(
        title={
            'text': "Distribution Summary",
            'font': dict(
                family='Raleway, sans-serif',
                size=22,
                color='rgba(0,0,0,0.85)'
            )
        },
        height=max(350, len(type_counts) * 30 + 100),
        margin=dict(l=10, r=10, t=50, b=15)
    )
    
    fig_table.write_html('outputs/type_distribution_table.html')
    print("Type distribution analysis saved to outputs/type_distribution.html and outputs/type_distribution_table.html")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze SERP type distribution')
    parser.add_argument('input_file', help='Path to the input CSV file')
    args = parser.parse_args()

    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    # Load the data
    print(f"\nLoading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)

    # Run the type distribution analysis
    print("\nRunning type distribution analysis...")
    analyze_type_distribution(df)

    print("\nAnalysis completed successfully!")
    print("Results have been saved to the 'outputs' directory.")

if __name__ == '__main__':
    main() 