"""
Visualization utilities module for PromptOpt Co-Pilot.

This module provides comprehensive chart generation, data visualization, and export
capabilities for optimization results, performance metrics, and analysis dashboards.
"""

import io
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Optional dependencies with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive charts will be disabled.")

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    warnings.warn("ReportLab not available. PDF export will be disabled.")

from backend.core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BOX = "box"
    HISTOGRAM = "histogram"
    PIE = "pie"
    AREA = "area"


class ExportFormat(Enum):
    """Supported export formats."""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


@dataclass
class TrendAnalysis:
    """Results from trend detection analysis."""
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    slope: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    seasonal_component: Optional[List[float]] = None
    residuals: Optional[List[float]] = None


@dataclass
class VisualizationConfig:
    """Configuration for chart styling and behavior."""
    theme: str = "default"  # 'default', 'dark', 'light', 'colorblind'
    color_palette: List[str] = field(default_factory=lambda: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 300
    font_family: str = "Arial"
    font_size: int = 12
    title_size: int = 14
    legend_position: str = "best"
    grid: bool = True
    style: str = "whitegrid"
    export_formats: List[str] = field(default_factory=lambda: ["png", "svg"])
    interactive: bool = True
    max_data_points: int = 10000  # For performance optimization


@dataclass
class ChartResult:
    """Container for generated chart with metadata."""
    figure: Optional[Figure]
    html_content: Optional[str]
    data: Dict[str, Any]
    chart_type: str
    title: str
    created_at: datetime
    export_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportResult:
    """Container for complete optimization report."""
    charts: List[ChartResult]
    summary_stats: Dict[str, Any]
    export_path: Optional[str]
    created_at: datetime
    report_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualizationManager:
    """Main class for managing chart generation and visualization."""
    
    def __init__(self, config: VisualizationConfig = None):
        """Initialize the visualization manager."""
        self.config = config or VisualizationConfig()
        self.settings = get_settings()
        self._setup_matplotlib()
        self._setup_seaborn()
        
    def _setup_matplotlib(self):
        """Configure matplotlib settings."""
        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['figure.dpi'] = self.config.dpi
        plt.rcParams['font.family'] = self.config.font_family
        plt.rcParams['font.size'] = self.config.font_size
        plt.rcParams['axes.titlesize'] = self.config.title_size
        plt.rcParams['axes.grid'] = self.config.grid
        
        if self.config.theme == "dark":
            plt.style.use('dark_background')
        elif self.config.theme == "light":
            plt.style.use('default')
            
    def _setup_seaborn(self):
        """Configure seaborn settings."""
        if self.config.theme == "dark":
            sns.set_theme(style="darkgrid", palette="bright")
        else:
            sns.set_theme(style=self.config.style, palette=self.config.color_palette)
    
    def create_optimization_progress_chart(self, data: List[Dict]) -> ChartResult:
        """Create a chart showing optimization progress over iterations."""
        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("No data provided for optimization progress chart")
                
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Plot main metric progression
            if 'iteration' in df.columns and 'best_score' in df.columns:
                ax.plot(df['iteration'], df['best_score'], 
                       marker='o', linewidth=2, markersize=4,
                       color=self.config.color_palette[0], label='Best Score')
                
            # Add current score if available
            if 'current_score' in df.columns:
                ax.plot(df['iteration'], df['current_score'], 
                       alpha=0.6, linewidth=1,
                       color=self.config.color_palette[1], label='Current Score')
            
            # Add moving average
            if len(df) > 5:
                window = min(5, len(df) // 4)
                moving_avg = df['best_score'].rolling(window=window, center=True).mean()
                ax.plot(df['iteration'], moving_avg, 
                       linestyle='--', linewidth=2,
                       color=self.config.color_palette[2], label=f'Moving Avg ({window})')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score')
            ax.set_title('Optimization Progress')
            ax.legend(loc=self.config.legend_position)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return ChartResult(
                figure=fig,
                html_content=None,
                data=data,
                chart_type=ChartType.LINE.value,
                title="Optimization Progress",
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating optimization progress chart: {e}")
            raise
    
    def create_variant_comparison_chart(self, variants: List[Dict], metrics: List[str]) -> ChartResult:
        """Create a comparison chart for different prompt variants."""
        try:
            df = pd.DataFrame(variants)
            if df.empty or not metrics:
                raise ValueError("No variants or metrics provided")
            
            # Prepare data for plotting
            n_metrics = len(metrics)
            n_variants = len(variants)
            
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                if metric not in df.columns:
                    continue
                    
                ax = axes[i]
                variant_names = df.get('name', [f'Variant {j+1}' for j in range(n_variants)])
                values = df[metric].values
                
                bars = ax.bar(range(len(values)), values, 
                             color=self.config.color_palette[:len(values)])
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.set_xlabel('Variants')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.set_xticks(range(len(variant_names)))
                ax.set_xticklabels(variant_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return ChartResult(
                figure=fig,
                html_content=None,
                data={'variants': variants, 'metrics': metrics},
                chart_type=ChartType.BAR.value,
                title="Variant Comparison",
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating variant comparison chart: {e}")
            raise
    
    def create_metric_distribution_chart(self, results: List[Dict], metric: str) -> ChartResult:
        """Create a distribution chart for a specific metric."""
        try:
            df = pd.DataFrame(results)
            if df.empty or metric not in df.columns:
                raise ValueError(f"No data or metric '{metric}' not found")
            
            values = df[metric].dropna()
            if len(values) == 0:
                raise ValueError(f"No valid values found for metric '{metric}'")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(values, bins='auto', alpha=0.7, 
                    color=self.config.color_palette[0], edgecolor='black')
            ax1.set_xlabel(metric.replace('_', ' ').title())
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = values.mean()
            std_val = values.std()
            ax1.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.3f}')
            ax1.axvline(mean_val + std_val, color='orange', linestyle='--', 
                       label=f'+1σ: {mean_val + std_val:.3f}')
            ax1.axvline(mean_val - std_val, color='orange', linestyle='--', 
                       label=f'-1σ: {mean_val - std_val:.3f}')
            ax1.legend()
            
            # Box plot
            ax2.boxplot(values, vert=True)
            ax2.set_ylabel(metric.replace('_', ' ').title())
            ax2.set_title(f'{metric.replace("_", " ").title()} Box Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return ChartResult(
                figure=fig,
                html_content=None,
                data=results,
                chart_type=ChartType.HISTOGRAM.value,
                title=f"{metric.replace('_', ' ').title()} Distribution",
                created_at=datetime.now(),
                metadata={'mean': mean_val, 'std': std_val, 'count': len(values)}
            )
            
        except Exception as e:
            logger.error(f"Error creating metric distribution chart: {e}")
            raise
    
    def create_performance_timeline(self, history: List[Dict]) -> ChartResult:
        """Create a timeline chart of performance history."""
        try:
            df = pd.DataFrame(history)
            if df.empty:
                raise ValueError("No history data provided")
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'created_at' in df.columns:
                df['timestamp'] = pd.to_datetime(df['created_at'])
            else:
                df['timestamp'] = pd.date_range(
                    start=datetime.now(), periods=len(df), freq='H'
                )
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Plot multiple metrics if available
            metrics = [col for col in df.columns if col not in ['timestamp', 'id', 'name']]
            colors = self.config.color_palette
            
            for i, metric in enumerate(metrics[:len(colors)]):
                if pd.api.types.is_numeric_dtype(df[metric]):
                    ax.plot(df['timestamp'], df[metric], 
                           marker='o', linewidth=2, markersize=3,
                           color=colors[i % len(colors)], label=metric.replace('_', ' ').title())
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Performance Score')
            ax.set_title('Performance Timeline')
            ax.legend(loc=self.config.legend_position)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df) // 10)))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            return ChartResult(
                figure=fig,
                html_content=None,
                data=history,
                chart_type=ChartType.LINE.value,
                title="Performance Timeline",
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating performance timeline: {e}")
            raise
    
    def create_correlation_matrix(self, metrics_data: Dict) -> ChartResult:
        """Create a correlation matrix heatmap for metrics."""
        try:
            df = pd.DataFrame(metrics_data)
            if df.empty:
                raise ValueError("No metrics data provided")
            
            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numeric columns for correlation")
            
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
            
            ax.set_title('Metrics Correlation Matrix')
            plt.tight_layout()
            
            return ChartResult(
                figure=fig,
                html_content=None,
                data=metrics_data,
                chart_type=ChartType.HEATMAP.value,
                title="Metrics Correlation Matrix",
                created_at=datetime.now(),
                metadata={'correlation_matrix': corr_matrix.to_dict()}
            )
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            raise
    
    def generate_optimization_report(self, results) -> ReportResult:
        """Generate a complete optimization report with multiple visualizations."""
        try:
            charts = []
            summary_stats = {}
            
            # Extract data from results
            if hasattr(results, 'history'):
                history_data = results.history
            else:
                history_data = results.get('history', [])
            
            if hasattr(results, 'variants'):
                variants_data = results.variants
            else:
                variants_data = results.get('variants', [])
            
            # Create progress chart
            if history_data:
                progress_chart = self.create_optimization_progress_chart(history_data)
                charts.append(progress_chart)
                
                # Calculate summary statistics
                df = pd.DataFrame(history_data)
                if 'best_score' in df.columns:
                    summary_stats['final_score'] = df['best_score'].iloc[-1]
                    summary_stats['initial_score'] = df['best_score'].iloc[0]
                    summary_stats['improvement'] = summary_stats['final_score'] - summary_stats['initial_score']
                    summary_stats['total_iterations'] = len(df)
            
            # Create variant comparison
            if variants_data:
                metrics = ['score', 'accuracy', 'relevance', 'coherence']
                available_metrics = [m for m in metrics if any(m in v for v in variants_data)]
                if available_metrics:
                    variant_chart = self.create_variant_comparison_chart(variants_data, available_metrics)
                    charts.append(variant_chart)
            
            # Create correlation matrix if enough data
            if len(variants_data) > 1:
                try:
                    corr_chart = self.create_correlation_matrix(variants_data)
                    charts.append(corr_chart)
                except Exception as e:
                    logger.warning(f"Could not create correlation matrix: {e}")
            
            return ReportResult(
                charts=charts,
                summary_stats=summary_stats,
                export_path=None,
                created_at=datetime.now(),
                report_type="optimization_report"
            )
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
            raise


# Chart generation functions
def plot_line_chart(data: List[Dict], x_axis: str, y_axis: str, **kwargs) -> Figure:
    """Generate a line chart from data."""
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    ax.plot(df[x_axis], df[y_axis], **{k: v for k, v in kwargs.items() if k != 'figsize'})
    ax.set_xlabel(x_axis.replace('_', ' ').title())
    ax.set_ylabel(y_axis.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_bar_chart(data: List[Dict], categories: str, values: str, **kwargs) -> Figure:
    """Generate a bar chart from data."""
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    ax.bar(df[categories], df[values], **{k: v for k, v in kwargs.items() if k != 'figsize'})
    ax.set_xlabel(categories.replace('_', ' ').title())
    ax.set_ylabel(values.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_scatter_plot(data: List[Dict], x_axis: str, y_axis: str, **kwargs) -> Figure:
    """Generate a scatter plot from data."""
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    ax.scatter(df[x_axis], df[y_axis], **{k: v for k, v in kwargs.items() if k != 'figsize'})
    ax.set_xlabel(x_axis.replace('_', ' ').title())
    ax.set_ylabel(y_axis.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_heatmap(data: List[List[float]], labels: List[str], **kwargs) -> Figure:
    """Generate a heatmap from 2D data."""
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
    
    im = ax.imshow(data, cmap=kwargs.get('cmap', 'viridis'), aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    plt.colorbar(im, ax=ax)
    return fig


def plot_box_plot(data: List[Dict], groupby: str, metric: str, **kwargs) -> Figure:
    """Generate a box plot grouped by category."""
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    groups = df.groupby(groupby)[metric].apply(list)
    ax.boxplot(groups.values, labels=groups.index)
    ax.set_xlabel(groupby.replace('_', ' ').title())
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_histogram(data: List[float], bins: int, **kwargs) -> Figure:
    """Generate a histogram from numeric data."""
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    ax.hist(data, bins=bins, **{k: v for k, v in kwargs.items() if k not in ['figsize', 'bins']})
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    return fig


# Export and format functions
def export_chart(figure: Figure, format_type: str, output_path: str) -> bool:
    """Export chart to specified format."""
    try:
        format_type = format_type.lower()
        
        if format_type == 'png':
            figure.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        elif format_type == 'svg':
            figure.savefig(output_path, format='svg', bbox_inches='tight')
        elif format_type == 'pdf':
            figure.savefig(output_path, format='pdf', bbox_inches='tight')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting chart: {e}")
        return False


def generate_interactive_chart(data: Dict, chart_type: str) -> str:
    """Generate interactive chart HTML using Plotly."""
    if not PLOTLY_AVAILABLE:
        return "<p>Interactive charts not available. Please install plotly.</p>"
    
    try:
        df = pd.DataFrame(data.get('data', []))
        
        if chart_type == 'line':
            fig = px.line(df, x=data.get('x'), y=data.get('y'))
        elif chart_type == 'bar':
            fig = px.bar(df, x=data.get('x'), y=data.get('y'))
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=data.get('x'), y=data.get('y'))
        else:
            return f"<p>Chart type '{chart_type}' not supported for interactive charts.</p>"
        
        return plot(fig, output_type='div', include_plotlyjs=True)
        
    except Exception as e:
        logger.error(f"Error generating interactive chart: {e}")
        return f"<p>Error generating chart: {e}</p>"


def create_dashboard_config(charts: List[ChartResult]) -> Dict:
    """Create configuration for frontend dashboard integration."""
    config = {
        'charts': [],
        'layout': {
            'columns': 2,
            'responsive': True
        },
        'theme': 'light',
        'created_at': datetime.now().isoformat()
    }
    
    for chart in charts:
        chart_config = {
            'id': f"chart_{len(config['charts'])}",
            'type': chart.chart_type,
            'title': chart.title,
            'data': chart.data,
            'metadata': chart.metadata
        }
        config['charts'].append(chart_config)
    
    return config


# Data processing functions
def prepare_optimization_data(raw_results: List[Dict]) -> Dict:
    """Transform raw optimization results for visualization."""
    df = pd.DataFrame(raw_results)
    
    processed = {
        'timeline': [],
        'metrics': {},
        'summary': {}
    }
    
    if 'iteration' in df.columns:
        processed['timeline'] = df.sort_values('iteration').to_dict('records')
    
    # Extract numeric metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        processed['metrics'][col] = {
            'values': df[col].dropna().tolist(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    processed['summary'] = {
        'total_experiments': len(df),
        'best_score': df.get('score', pd.Series()).max() if 'score' in df.columns else None,
        'completion_rate': (df['status'] == 'completed').mean() if 'status' in df.columns else 1.0
    }
    
    return processed


def aggregate_metrics(results: List[Dict], groupby: str) -> Dict:
    """Aggregate metrics by specified grouping."""
    df = pd.DataFrame(results)
    
    if groupby not in df.columns:
        raise ValueError(f"Grouping column '{groupby}' not found")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    aggregated = df.groupby(groupby)[numeric_cols].agg(['mean', 'std', 'count']).round(3)
    
    return aggregated.to_dict()


def calculate_statistical_summaries(data: List[float]) -> Dict:
    """Calculate statistical summaries for chart annotations."""
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    if len(data) == 0:
        return {}
    
    return {
        'count': len(data),
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data))
    }


def detect_trends(timeseries_data: List[Dict]) -> TrendAnalysis:
    """Detect trends in time series data."""
    df = pd.DataFrame(timeseries_data)
    
    if 'value' not in df.columns:
        raise ValueError("Time series data must contain 'value' column")
    
    # Prepare data
    y = df['value'].dropna().values
    x = np.arange(len(y))
    
    if len(y) < 3:
        return TrendAnalysis(
            trend_direction='insufficient_data',
            slope=0.0,
            r_squared=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0)
        )
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Determine trend direction
    if p_value < 0.05:  # Significant trend
        if slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
    else:
        trend_direction = 'stable'
    
    # Confidence interval for slope
    confidence_interval = (slope - 1.96 * std_err, slope + 1.96 * std_err)
    
    return TrendAnalysis(
        trend_direction=trend_direction,
        slope=slope,
        r_squared=r_value**2,
        p_value=p_value,
        confidence_interval=confidence_interval
    )


def identify_outliers(data: List[float], method: str = 'iqr') -> List[int]:
    """Identify outlier indices in data."""
    data = np.array(data)
    outlier_indices = []
    
    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
        
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        outlier_indices = np.where(z_scores > 3)[0].tolist()
        
    elif method == 'isolation_forest':
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data.reshape(-1, 1))
            outlier_indices = np.where(outliers == -1)[0].tolist()
        except ImportError:
            logger.warning("sklearn not available, falling back to IQR method")
            return identify_outliers(data, method='iqr')
    
    return outlier_indices


# Specialized optimization visualizations
def create_convergence_plot(optimization_history: List[Dict]) -> ChartResult:
    """Create convergence plot for optimization algorithms."""
    df = pd.DataFrame(optimization_history)
    
    if 'iteration' not in df.columns or 'best_score' not in df.columns:
        raise ValueError("Optimization history must contain 'iteration' and 'best_score' columns")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main convergence plot
    ax1.plot(df['iteration'], df['best_score'], 'b-', linewidth=2, label='Best Score')
    
    if 'current_score' in df.columns:
        ax1.plot(df['iteration'], df['current_score'], 'r-', alpha=0.6, 
                linewidth=1, label='Current Score')
    
    # Add convergence threshold if available
    if len(df) > 10:
        # Calculate rate of improvement
        window = min(10, len(df) // 4)
        rolling_improvement = df['best_score'].diff().rolling(window=window).mean()
        convergence_threshold = rolling_improvement.std() * 0.1
        
        # Mark convergence point
        converged_idx = None
        for i in range(window, len(rolling_improvement)):
            if abs(rolling_improvement.iloc[i]) < convergence_threshold:
                converged_idx = i
                break
        
        if converged_idx:
            ax1.axvline(x=df['iteration'].iloc[converged_idx], color='green', 
                       linestyle='--', alpha=0.7, label=f'Convergence (iter {converged_idx})')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Score')
    ax1.set_title('Optimization Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement rate plot
    if len(df) > 1:
        improvement = df['best_score'].diff()
        ax2.plot(df['iteration'][1:], improvement[1:], 'g-', linewidth=2, label='Score Improvement')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Score Improvement')
        ax2.set_title('Rate of Improvement')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return ChartResult(
        figure=fig,
        html_content=None,
        data=optimization_history,
        chart_type="convergence",
        title="Optimization Convergence Analysis",
        created_at=datetime.now(),
        metadata={'converged_at': converged_idx if 'converged_idx' in locals() else None}
    )


def create_pareto_front_chart(multi_objective_results: List[Dict]) -> ChartResult:
    """Create Pareto front visualization for multi-objective optimization."""
    df = pd.DataFrame(multi_objective_results)
    
    # Identify objective columns (assume they start with 'obj_' or contain 'objective')
    obj_cols = [col for col in df.columns if 'obj' in col.lower() or 'objective' in col.lower()]
    
    if len(obj_cols) < 2:
        raise ValueError("Need at least 2 objectives for Pareto front visualization")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # For 2D Pareto front
    if len(obj_cols) >= 2:
        x_obj, y_obj = obj_cols[0], obj_cols[1]
        
        # Plot all solutions
        ax.scatter(df[x_obj], df[y_obj], alpha=0.6, s=50, 
                  color='lightblue', label='All Solutions')
        
        # Identify Pareto front solutions
        pareto_mask = _identify_pareto_front(df[obj_cols].values)
        pareto_solutions = df[pareto_mask]
        
        # Plot Pareto front
        ax.scatter(pareto_solutions[x_obj], pareto_solutions[y_obj], 
                  color='red', s=80, alpha=0.8, label='Pareto Front', zorder=5)
        
        # Connect Pareto front points
        pareto_sorted = pareto_solutions.sort_values(x_obj)
        ax.plot(pareto_sorted[x_obj], pareto_sorted[y_obj], 
               'r--', alpha=0.7, linewidth=2, zorder=4)
        
        ax.set_xlabel(x_obj.replace('_', ' ').title())
        ax.set_ylabel(y_obj.replace('_', ' ').title())
        ax.set_title('Pareto Front Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return ChartResult(
        figure=fig,
        html_content=None,
        data=multi_objective_results,
        chart_type="pareto_front",
        title="Pareto Front Analysis",
        created_at=datetime.now(),
        metadata={'pareto_solutions': len(pareto_solutions) if 'pareto_solutions' in locals() else 0}
    )


def create_parameter_sensitivity_chart(sensitivity_data: Dict) -> ChartResult:
    """Create parameter sensitivity analysis chart."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    parameters = list(sensitivity_data.keys())[:4]  # Limit to 4 parameters
    
    for i, param in enumerate(parameters):
        ax = axes[i]
        param_data = sensitivity_data[param]
        
        if 'values' in param_data and 'scores' in param_data:
            values = param_data['values']
            scores = param_data['scores']
            
            # Main sensitivity plot
            ax.plot(values, scores, 'bo-', linewidth=2, markersize=6)
            
            # Highlight optimal value
            best_idx = np.argmax(scores)
            ax.plot(values[best_idx], scores[best_idx], 'ro', markersize=10, 
                   label=f'Optimal: {values[best_idx]:.3f}')
            
            # Add confidence intervals if available
            if 'confidence_lower' in param_data and 'confidence_upper' in param_data:
                ax.fill_between(values, param_data['confidence_lower'], 
                              param_data['confidence_upper'], alpha=0.3)
            
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Performance Score')
            ax.set_title(f'{param.replace("_", " ").title()} Sensitivity')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(parameters), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return ChartResult(
        figure=fig,
        html_content=None,
        data=sensitivity_data,
        chart_type="sensitivity",
        title="Parameter Sensitivity Analysis",
        created_at=datetime.now()
    )


def create_ablation_study_chart(ablation_results: List[Dict]) -> ChartResult:
    """Create ablation study visualization."""
    df = pd.DataFrame(ablation_results)
    
    if 'component' not in df.columns or 'score' not in df.columns:
        raise ValueError("Ablation results must contain 'component' and 'score' columns")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Component importance (difference from baseline)
    baseline_score = df[df['component'] == 'baseline']['score'].iloc[0] if 'baseline' in df['component'].values else df['score'].max()
    df['importance'] = baseline_score - df['score']
    
    # Bar chart of component importance
    components = df[df['component'] != 'baseline']['component'] if 'baseline' in df['component'].values else df['component']
    importance_scores = df[df['component'] != 'baseline']['importance'] if 'baseline' in df['component'].values else df['importance']
    
    bars = ax1.bar(range(len(components)), importance_scores, 
                   color=['red' if x > 0 else 'green' for x in importance_scores])
    
    ax1.set_xlabel('Components')
    ax1.set_ylabel('Performance Drop (Importance)')
    ax1.set_title('Component Importance (Ablation Study)')
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(components, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, importance_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Cumulative importance
    cumulative_importance = np.cumsum(sorted(importance_scores, reverse=True))
    ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
            'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Components Removed')
    ax2.set_ylabel('Cumulative Performance Drop')
    ax2.set_title('Cumulative Impact Analysis')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return ChartResult(
        figure=fig,
        html_content=None,
        data=ablation_results,
        chart_type="ablation",
        title="Ablation Study Analysis",
        created_at=datetime.now(),
        metadata={'most_important_component': components[np.argmax(importance_scores)]}
    )


def _identify_pareto_front(objectives: np.ndarray) -> np.ndarray:
    """Identify Pareto front solutions from multi-objective data."""
    n_points = objectives.shape[0]
    pareto_front = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i (assuming maximization)
                if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                    pareto_front[i] = False
                    break
    
    return pareto_front


def export_report_pdf(report_data: Dict, template: str) -> bytes:
    """Generate PDF report from report data."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab not available for PDF generation")
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph(report_data.get('title', 'Optimization Report'), styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Summary statistics
        if 'summary_stats' in report_data:
            summary_title = Paragraph('Summary Statistics', styles['Heading1'])
            story.append(summary_title)
            
            for key, value in report_data['summary_stats'].items():
                text = f"<b>{key.replace('_', ' ').title()}:</b> {value}"
                para = Paragraph(text, styles['Normal'])
                story.append(para)
            
            story.append(Spacer(1, 12))
        
        # Charts (would need to be saved as images first)
        if 'charts' in report_data:
            charts_title = Paragraph('Visualizations', styles['Heading1'])
            story.append(charts_title)
            
            for chart in report_data['charts']:
                chart_title = Paragraph(chart.get('title', 'Chart'), styles['Heading2'])
                story.append(chart_title)
                # Note: In practice, you'd save the matplotlib figures as images
                # and insert them here using the Image class
                story.append(Spacer(1, 12))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        raise


# Utility functions for advanced visualizations
def create_interactive_dashboard(charts: List[ChartResult]) -> str:
    """Create an interactive HTML dashboard with multiple charts."""
    if not PLOTLY_AVAILABLE:
        return "<p>Interactive dashboards require Plotly. Please install plotly.</p>"
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PromptOpt Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .chart-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .dashboard-title { text-align: center; color: #333; margin-bottom: 30px; }
        </style>
    </head>
    <body>
        <h1 class="dashboard-title">PromptOpt Optimization Dashboard</h1>
        {chart_divs}
    </body>
    </html>
    """
    
    chart_divs = ""
    for i, chart in enumerate(charts):
        if chart.html_content:
            chart_divs += f'<div class="chart-container"><h2>{chart.title}</h2>{chart.html_content}</div>\n'
    
    return html_template.format(chart_divs=chart_divs)


def optimize_chart_performance(data: List[Dict], max_points: int = 1000) -> List[Dict]:
    """Optimize chart data for performance by sampling large datasets."""
    if len(data) <= max_points:
        return data
    
    # Use systematic sampling to preserve trends
    step = len(data) // max_points
    sampled_data = data[::step]
    
    # Always include first and last points
    if data[0] not in sampled_data:
        sampled_data.insert(0, data[0])
    if data[-1] not in sampled_data:
        sampled_data.append(data[-1])
    
    logger.info(f"Sampled {len(sampled_data)} points from {len(data)} for performance")
    return sampled_data


# Test utilities (for development and debugging)
def generate_sample_data(chart_type: str, n_points: int = 100) -> List[Dict]:
    """Generate sample data for testing chart functions."""
    np.random.seed(42)  # For reproducible results
    
    if chart_type == "optimization_progress":
        return [
            {
                'iteration': i,
                'best_score': 0.5 + 0.4 * (1 - np.exp(-i/20)) + np.random.normal(0, 0.02),
                'current_score': 0.3 + 0.5 * np.random.random() + np.random.normal(0, 0.05)
            }
            for i in range(n_points)
        ]
    
    elif chart_type == "variants":
        variants = ['Baseline', 'Variant A', 'Variant B', 'Variant C', 'Variant D']
        return [
            {
                'name': variant,
                'score': np.random.uniform(0.6, 0.95),
                'accuracy': np.random.uniform(0.7, 0.98),
                'relevance': np.random.uniform(0.65, 0.92),
                'coherence': np.random.uniform(0.75, 0.96)
            }
            for variant in variants
        ]
    
    elif chart_type == "metrics_distribution":
        return [
            {'score': np.random.normal(0.8, 0.1)}
            for _ in range(n_points)
        ]
    
    else:
        return [{'x': i, 'y': np.random.random()} for i in range(n_points)]


if __name__ == "__main__":
    # Example usage and testing
    config = VisualizationConfig(theme="light", interactive=True)
    viz_manager = VisualizationManager(config)
    
    # Generate sample data
    progress_data = generate_sample_data("optimization_progress", 50)
    variants_data = generate_sample_data("variants")
    
    try:
        # Create charts
        progress_chart = viz_manager.create_optimization_progress_chart(progress_data)
        variants_chart = viz_manager.create_variant_comparison_chart(
            variants_data, ['score', 'accuracy', 'relevance']
        )
        
        # Export charts
        export_chart(progress_chart.figure, 'png', 'progress_chart.png')
        export_chart(variants_chart.figure, 'svg', 'variants_chart.svg')
        
        print("Charts generated successfully!")
        
        # Generate complete report
        mock_results = type('MockResults', (), {
            'history': progress_data,
            'variants': variants_data
        })()
        
        report = viz_manager.generate_optimization_report(mock_results)
        print(f"Generated report with {len(report.charts)} charts")
        
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
        print(f"Error: {e}")