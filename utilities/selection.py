import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class MetricDirection(Enum):
    LOWER_IS_BETTER = "lower"
    HIGHER_IS_BETTER = "higher"

@dataclass
class ModelScore:
    model_name: str
    approach: str
    total_score: float
    metric_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    rank: int = 0

class ModelSelector:
    def __init__(self, metrics_config: Dict[str, Dict]):
        """
        Initialize model selector with metrics configuration
        
        Parameters:
        -----------
        metrics_config : dict
            Dictionary with metric names as keys and config as values
            Example: {'r2': {'direction': 'higher', 'weight': 1.0}, 
                     'rmse': {'direction': 'lower', 'weight': 1.0}}
        """
        self.metrics_config = metrics_config
        self.metric_directions = {}
        self.metric_weights = {}
        
        for metric, config in metrics_config.items():
            direction = config.get('better_direction', 'lower')
            self.metric_directions[metric] = MetricDirection.LOWER_IS_BETTER if direction == 'lower' else MetricDirection.HIGHER_IS_BETTER
            self.metric_weights[metric] = config.get('weight', 1.0)
    
    def normalize_metrics(self, results: Dict[str, Dict], selected_metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Normalize metrics across all models to 0-1 scale
        
        Parameters:
        -----------
        results : dict
            Model results dictionary
        selected_metrics : list
            List of metrics to consider
            
        Returns:
        --------
        dict
            Normalized metric values for each model
        """
        # Extract metric values for all models
        metric_values = {metric: [] for metric in selected_metrics}
        model_names = []
        
        for model_key, model_result in results.items():
            if 'error' in model_result:
                continue
            if 'approach3' in model_key:
                continue
                
            model_names.append(model_key)
            
            # Handle different result structures for different approaches
            if 'stats' in model_result:
                if isinstance(model_result['stats'], list):
                    # Approach 2: average metrics across datasets
                    for metric in selected_metrics:
                        if metric in model_result['stats'][0].columns:
                            avg_value = np.mean([
                                model_result['stats'][0][metric].iloc[0],
                                model_result['stats'][1][metric].iloc[0]
                            ])
                            metric_values[metric].append(avg_value)
                        else:
                            metric_values[metric].append(np.nan)
                else:
                    # Approach 1: single dataset
                    for metric in selected_metrics:
                        if metric in model_result['stats'].columns:
                            metric_values[metric].append(model_result['stats'][metric].iloc[0])
                        else:
                            metric_values[metric].append(np.nan)
        
        # Normalize each metric
        normalized_results = {}
        
        for model_name in model_names:
            normalized_results[model_name] = {}
        
        for metric in selected_metrics:
            values = np.array(metric_values[metric])
            valid_mask = ~np.isnan(values)
            
            if not np.any(valid_mask):
                # All values are NaN
                for model_name in model_names:
                    normalized_results[model_name][metric] = 0.0
                continue
            
            valid_values = values[valid_mask]
            
            if len(np.unique(valid_values)) == 1:
                # All valid values are the same
                for model_name in model_names:
                    normalized_results[model_name][metric] = 1.0 if not np.isnan(values[model_names.index(model_name)]) else 0.0
                continue
            
            # Normalize based on direction
            if self.metric_directions[metric] == MetricDirection.HIGHER_IS_BETTER:
                # For metrics where higher is better (e.g., R²)
                min_val, max_val = np.min(valid_values), np.max(valid_values)
                normalized = (values - min_val) / (max_val - min_val)
            else:
                # For metrics where lower is better (e.g., RMSE, AIC)
                min_val, max_val = np.min(valid_values), np.max(valid_values)
                normalized = 1 - (values - min_val) / (max_val - min_val)
            
            # Handle NaN values
            normalized = np.where(np.isnan(normalized), 0.0, normalized)
            
            for i, model_name in enumerate(model_names):
                normalized_results[model_name][metric] = normalized[i]
        
        return normalized_results
    
    def calculate_composite_score(self, normalized_metrics: Dict[str, float], 
                                selected_metrics: List[str], 
                                method: str = 'weighted_average') -> float:
        """
        Calculate composite score from normalized metrics
        
        Parameters:
        -----------
        normalized_metrics : dict
            Normalized metric values
        selected_metrics : list
            List of metrics to include
        method : str
            Scoring method ('weighted_average', 'geometric_mean', 'harmonic_mean')
            
        Returns:
        --------
        float
            Composite score
        """
        if method == 'weighted_average':
            total_weight = sum(self.metric_weights.get(metric, 1.0) for metric in selected_metrics)
            if total_weight == 0:
                return 0.0
            
            weighted_sum = sum(
                normalized_metrics.get(metric, 0.0) * self.metric_weights.get(metric, 1.0)
                for metric in selected_metrics
            )
            return weighted_sum / total_weight
        
        elif method == 'geometric_mean':
            values = [normalized_metrics.get(metric, 0.0) for metric in selected_metrics]
            # Add small epsilon to avoid log(0)
            values = [max(v, 1e-10) for v in values]
            return np.exp(np.mean(np.log(values)))
        
        elif method == 'harmonic_mean':
            values = [normalized_metrics.get(metric, 0.0) for metric in selected_metrics]
            # Add small epsilon to avoid division by 0
            values = [max(v, 1e-10) for v in values]
            return len(values) / sum(1/v for v in values)
        
        else:
            raise ValueError(f"Unknown scoring method: {method}")
    
    def identify_best_models(self, results: Dict[str, Dict], 
                           selected_metrics: List[str],
                           approaches: Dict[str, Dict],
                           scoring_method: str = 'weighted_average',
                           top_n: int = 5) -> Tuple[List[ModelScore], pd.DataFrame]:
        """
        Identify the best models across all approaches
        
        Parameters:
        -----------
        results : dict
            Model results dictionary
        selected_metrics : list
            List of metrics to consider
        approaches : dict
            Approaches configuration
        scoring_method : str
            Method for calculating composite scores
        top_n : int
            Number of top models to return
            
        Returns:
        --------
        tuple
            (List of ModelScore objects, Summary DataFrame)
        """
        # Normalize metrics
        normalized_metrics = self.normalize_metrics(results, selected_metrics)
        
        # Calculate scores for each model
        model_scores = []
        
        for model_key, normalized_vals in normalized_metrics.items():
            # Determine approach and model name
            if ':' in model_key:
                approach_id, model_name = model_key.split(':', 1)
            else:
                approach_id = model_key
                model_name = model_key
            
            # Calculate composite score
            composite_score = self.calculate_composite_score(
                normalized_vals, selected_metrics, scoring_method
            )
            
            model_score = ModelScore(
                model_name=model_name,
                approach=approaches.get(approach_id, {}).get('display_name', approach_id),
                total_score=composite_score,
                metric_scores={},
                normalized_scores=normalized_vals
            )
            
            # Add original metric scores
            if model_key in results and 'stats' in results[model_key]:
                stats = results[model_key]['stats']
                if isinstance(stats, list):
                    # Average across datasets for approach 2
                    for metric in selected_metrics:
                        if metric in stats[0].columns:
                            avg_value = np.mean([
                                stats[0][metric].iloc[0],
                                stats[1][metric].iloc[0]
                            ])
                            model_score.metric_scores[metric] = avg_value
                else:
                    # Single dataset
                    for metric in selected_metrics:
                        if metric in stats.columns:
                            model_score.metric_scores[metric] = stats[metric].iloc[0]
            
            model_scores.append(model_score)
        
        # Sort by composite score (descending)
        model_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Assign ranks
        for i, model_score in enumerate(model_scores):
            model_score.rank = i + 1
        
        # Create summary DataFrame
        summary_data = []
        for model_score in model_scores[:top_n]:
            row = {
                'Rank': model_score.rank,
                'Model': model_score.model_name,
                'Approach': model_score.approach,
                'Composite Score': model_score.total_score
            }
            
            # Add original metric values
            for metric in selected_metrics:
                if metric in model_score.metric_scores:
                    row[f'{metric}'] = model_score.metric_scores[metric]
            
            # Add normalized scores
            for metric in selected_metrics:
                if metric in model_score.normalized_scores:
                    row[f'{metric}_normalized'] = model_score.normalized_scores[metric]
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        return model_scores[:top_n], summary_df
    
    def get_best_model_by_approach(self, results: Dict[str, Dict], 
                                 selected_metrics: List[str],
                                 approaches: Dict[str, Dict],
                                 scoring_method: str = 'weighted_average') -> Dict[str, ModelScore]:
        """
        Get the best model for each approach
        
        Returns:
        --------
        dict
            Dictionary with approach names as keys and best ModelScore as values
        """
        all_scores, _ = self.identify_best_models(
            results, selected_metrics, approaches, scoring_method, top_n=100
        )
        
        best_by_approach = {}
        
        for score in all_scores:
            approach = score.approach
            if approach not in best_by_approach or score.total_score > best_by_approach[approach].total_score:
                best_by_approach[approach] = score
        
        return best_by_approach
    
    def create_model_comparison_report(self, results: Dict[str, Dict], 
                                     selected_metrics: List[str],
                                     approaches: Dict[str, Dict],
                                     scoring_method: str = 'weighted_average') -> str:
        """
        Create a comprehensive model comparison report
        
        Returns:
        --------
        str
            HTML formatted report
        """
        top_models, summary_df = self.identify_best_models(
            results, selected_metrics, approaches, scoring_method
        )
        
        best_by_approach = self.get_best_model_by_approach(
            results, selected_metrics, approaches, scoring_method
        )
        
        # Generate HTML report
        html = f"""
        <div class="model-selection-report">
            <h3>Model Selection Report</h3>
            
            <h4>Overall Best Models</h4>
            <p><strong>Best Model:</strong> {top_models[0].model_name} ({top_models[0].approach}) 
               - Score: {top_models[0].total_score:.4f}</p>
            
            <h4>Best Model by Approach</h4>
            <ul>
        """
        
        for approach, model_score in best_by_approach.items():
            html += f"<li><strong>{approach}:</strong> {model_score.model_name} (Score: {model_score.total_score:.4f})</li>"
        
        html += """
            </ul>
            
            <h4>Scoring Method</h4>
            <p>Composite scores calculated using: """ + scoring_method.replace('_', ' ').title() + """</p>
            
            <h4>Metric Weights</h4>
            <ul>
        """
        
        for metric, weight in self.metric_weights.items():
            if metric in selected_metrics:
                direction = "↑" if self.metric_directions[metric] == MetricDirection.HIGHER_IS_BETTER else "↓"
                html += f"<li>{self.metrics_config[metric]['display_name']}: {weight} {direction}</li>"
        
        html += """
            </ul>
        </div>
        """
        
        return html

# Usage example and integration with your existing code
def create_model_selector_from_metrics(metrics: Dict[str, Dict]) -> ModelSelector:
    """
    Create ModelSelector from your existing metrics configuration
    
    Parameters:
    -----------
    metrics : dict
        Your existing metrics dictionary
        
    Returns:
    --------
    ModelSelector
        Configured model selector
    """
    metrics_config = {}
    
    for metric_name, metric_info in metrics.items():
        config = {
            'display_name': metric_info['display_name'],
            'better_direction': metric_info.get('better_direction', 'lower'),
            'weight': 1.0  # Default weight, can be customized
        }
        metrics_config[metric_name] = config
    
    return ModelSelector(metrics_config)

# Integration function for your existing workflow
def select_best_models(results: Dict[str, Dict], 
                      selected_metrics: List[str],
                      approaches: Dict[str, Dict],
                      metrics: Dict[str, Dict],
                      method: str = 'weighted_average') -> Tuple[List[ModelScore], str]:
    """
    Main function to integrate with your existing code
    
    Parameters:
    -----------
    results : dict
        Your model results
    selected_metrics : list
        User-selected metrics
    approaches : dict
        Your approaches configuration
    metrics : dict
        Your metrics configuration
    method : str
        Scoring method
        
    Returns:
    --------
    tuple
        (Best models list, HTML report)
    """
    # Create model selector
    selector = create_model_selector_from_metrics(metrics)
    
    # Get best models
    best_models, summary_df = selector.identify_best_models(
        results, selected_metrics, approaches, method
    )
    
    # Generate report
    report = selector.create_model_comparison_report(
        results, selected_metrics, approaches, method
    )
    
    return best_models, report

if 0:
# Example usage in your app.py
    def enhanced_model_comparison(results, selected_metrics, approaches, metrics):
        """
        Enhanced model comparison with automatic best model selection
        """
        # Your existing comparison code...
        comparisons, ks_tables = {}, {}
        for approach in selected_approaches:
            comparisons[approach], ks_tables[approach] = create_comparison(results, approach)
        
        # Add automatic model selection
        best_models, selection_report = select_best_models(
            results, selected_metrics, approaches, metrics
        )
        
        return comparisons, ks_tables, best_models, selection_report

