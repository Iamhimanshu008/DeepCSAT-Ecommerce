import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_csat_distribution(df, target_col='CSAT Score'):
    if target_col in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=target_col, hue=target_col,
                      palette='Set2', ax=ax, legend=False)
        ax.set_title("CSAT Score Distribution")
        return fig
    return None


def plot_avg_resolution_vs_csat(df):
    if 'resolution_time' not in df.columns and 'connected_handling_time' in df.columns:
        df['resolution_time'] = df['connected_handling_time']

    if 'resolution_time' in df.columns and 'CSAT Score' in df.columns:
        grouped = df.groupby('CSAT Score')[
            'resolution_time'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='CSAT Score', y='resolution_time',
                    data=grouped, palette='coolwarm', ax=ax)
        ax.set_title("Avg Resolution Time by CSAT Score")
        return fig
    return None


def plot_channel_vs_csat(df):
    if 'channel_name' in df.columns and 'CSAT Score' in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='channel_name', y='CSAT Score',
                    hue='channel_name', palette='pastel', ax=ax, legend=False)
        ax.set_title("CSAT Score by Support Channel")
        return fig
    return None
