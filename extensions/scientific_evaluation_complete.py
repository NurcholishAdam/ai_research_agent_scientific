# -*- coding: utf-8 -*-
"""
Complete Scientific Research Evaluation Framework
Comprehensive evaluation for scientific research capabilities
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class ScientificMetrics:
    """Scientific evaluation metrics"""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    novelty_coverage: float = 0.0
    impact_coverage: float = 0.0
    citation_diversity: float = 0.0
    temporal_diversity: float = 0.0
    method_diversity: float = 0.0
    domain_relevance: float = 0.0

class ScientificEvaluator:
    """Complete scientific research evaluator"""
    
    def __init__(self):
        self.evaluation_history = []
        self.baseline_metrics = {}
    
    def evaluate_paper_retrieval(self, query: str, retrieved_papers: List[Dict], 
                                ground_truth: List[Dict]) -> ScientificMetrics:
        """Evaluate scientific paper retrieval"""
        
        # Basic retrieval metrics
        retrieved_ids = {p.get("paper_id", p.get("title", "")) for p in retrieved_papers}
        relevant_ids = {p.get("paper_id", p.get("title", "")) for p in ground_truth}
        
        if not retrieved_ids or not relevant_ids:
            return ScientificMetrics()
        
        intersection = retrieved_ids.intersection(relevant_ids)
        
        precision = len(intersection) / len(retrieved_ids)
        recall = len(intersection) / len(relevant_ids)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Scientific-specific metrics
        novelty_coverage = self._calculate_novelty_coverage(retrieved_papers)
        impact_coverage = self._calculate_impact_coverage(retrieved_papers)
        citation_diversity = self._calculate_citation_diversity(retrieved_papers)
        temporal_diversity = self._calculate_temporal_diversity(retrieved_papers)
        method_diversity = self._calculate_method_diversity(retrieved_papers)
        domain_relevance = self._calculate_domain_relevance(retrieved_papers, query)
        
        return ScientificMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            novelty_coverage=novelty_coverage,
            impact_coverage=impact_coverage,
            citation_diversity=citation_diversity,
            temporal_diversity=temporal_diversity,
            method_diversity=method_diversity,
            domain_relevance=domain_relevance
        )
    
    def evaluate_novelty_detection(self, papers: List[Dict], 
                                  novelty_predictions: List[float]) -> Dict[str, float]:
        """Evaluate novelty detection accuracy"""
        
        if len(papers) != len(novelty_predictions):
            return {"error": "Mismatched paper and prediction counts"}
        
        # Mock ground truth based on publication year (newer = more novel)
        current_year = datetime.now().year
        ground_truth = []
        
        for paper in papers:
            year = paper.get("year", current_year)
            # Papers from last 2 years considered novel
            novelty_gt = 1.0 if year >= current_year - 2 else 0.3
            ground_truth.append(novelty_gt)
        
        # Convert to binary classification
        threshold = 0.7
        pred_binary = [1 if p > threshold else 0 for p in novelty_predictions]
        gt_binary = [1 if g > threshold else 0 for g in ground_truth]
        
        # Calculate metrics
        tp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 0 and g == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "novelty_precision": precision,
            "novelty_recall": recall,
            "novelty_f1": f1,
            "novelty_accuracy": sum(1 for p, g in zip(pred_binary, gt_binary) if p == g) / len(pred_binary)
        }
    
    def evaluate_impact_prediction(self, papers: List[Dict], 
                                  impact_predictions: List[float]) -> Dict[str, float]:
        """Evaluate impact prediction accuracy"""
        
        if len(papers) != len(impact_predictions):
            return {"error": "Mismatched paper and prediction counts"}
        
        # Use actual citation counts as ground truth
        actual_citations = [paper.get("citations", 0) for paper in papers]
        
        # Normalize citations to 0-1 scale
        max_citations = max(actual_citations) if actual_citations else 1
        normalized_citations = [c / max_citations for c in actual_citations]
        
        # Calculate correlation
        if len(impact_predictions) > 1:
            correlation = self._calculate_correlation(impact_predictions, normalized_citations)
        else:
            correlation = 0.0
        
        # Calculate accuracy for high-impact papers
        threshold = 0.7
        pred_high_impact = [p > threshold for p in impact_predictions]
        actual_high_impact = [c > threshold for c in normalized_citations]
        
        accuracy = sum(1 for p, a in zip(pred_high_impact, actual_high_impact) if p == a) / len(pred_high_impact)
        
        return {
            "impact_correlation": correlation,
            "impact_accuracy": accuracy,
            "high_impact_precision": self._calculate_precision(pred_high_impact, actual_high_impact),
            "high_impact_recall": self._calculate_recall(pred_high_impact, actual_high_impact)
        }
    
    def evaluate_trend_analysis(self, trend_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate research trend analysis"""
        
        # Mock evaluation - in real scenario would compare against expert annotations
        trend_accuracy = 0.8  # How accurate are the identified trends
        emerging_detection = 0.75  # How well does it detect emerging areas
        temporal_analysis = 0.85  # How well does it analyze temporal patterns
        
        # Check if results contain expected components
        has_top_methods = "top_methods" in trend_results
        has_emerging = "emerging_methods" in trend_results.get("trends", {})
        has_temporal = "year_distribution" in trend_results.get("trends", {})
        
        completeness = sum([has_top_methods, has_emerging, has_temporal]) / 3.0
        
        return {
            "trend_accuracy": trend_accuracy,
            "emerging_detection": emerging_detection,
            "temporal_analysis": temporal_analysis,
            "analysis_completeness": completeness
        }
    
    def evaluate_research_gaps(self, identified_gaps: List[str], 
                              ground_truth_gaps: List[str] = None) -> Dict[str, float]:
        """Evaluate research gap identification"""
        
        if ground_truth_gaps is None:
            # Mock ground truth gaps
            ground_truth_gaps = [
                "Limited multilingual evaluation",
                "Insufficient human evaluation studies",
                "Limited interdisciplinary research",
                "Lack of ethical considerations",
                "Limited reproducibility studies"
            ]
        
        # Calculate overlap
        identified_set = set(gap.lower() for gap in identified_gaps)
        ground_truth_set = set(gap.lower() for gap in ground_truth_gaps)
        
        intersection = identified_set.intersection(ground_truth_set)
        
        precision = len(intersection) / len(identified_set) if identified_set else 0.0
        recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "gap_detection_precision": precision,
            "gap_detection_recall": recall,
            "gap_detection_f1": f1,
            "gaps_identified": len(identified_gaps)
        }
    
    def _calculate_novelty_coverage(self, papers: List[Dict]) -> float:
        """Calculate novelty coverage in retrieved papers"""
        if not papers:
            return 0.0
        
        novel_papers = 0
        for paper in papers:
            # Simple heuristic: papers with "novel", "new", "improved" in title
            title = paper.get("title", "").lower()
            if any(keyword in title for keyword in ["novel", "new", "improved", "enhanced"]):
                novel_papers += 1
        
        return novel_papers / len(papers)
    
    def _calculate_impact_coverage(self, papers: List[Dict]) -> float:
        """Calculate impact coverage in retrieved papers"""
        if not papers:
            return 0.0
        
        high_impact_papers = 0
        for paper in papers:
            citations = paper.get("citations", 0)
            # Consider papers with >1000 citations as high impact
            if citations > 1000:
                high_impact_papers += 1
        
        return high_impact_papers / len(papers)
    
    def _calculate_citation_diversity(self, papers: List[Dict]) -> float:
        """Calculate citation diversity"""
        if not papers:
            return 0.0
        
        citation_counts = [paper.get("citations", 0) for paper in papers]
        unique_citations = len(set(citation_counts))
        
        return unique_citations / len(citation_counts)
    
    def _calculate_temporal_diversity(self, papers: List[Dict]) -> float:
        """Calculate temporal diversity"""
        if not papers:
            return 0.0
        
        years = [paper.get("year", 2024) for paper in papers]
        unique_years = len(set(years))
        
        return min(unique_years / 10.0, 1.0)  # Normalize to max 10 years
    
    def _calculate_method_diversity(self, papers: List[Dict]) -> float:
        """Calculate methodological diversity"""
        if not papers:
            return 0.0
        
        all_methods = []
        for paper in papers:
            methods = paper.get("methodology", [])
            all_methods.extend(methods)
        
        if not all_methods:
            return 0.0
        
        unique_methods = len(set(all_methods))
        return min(unique_methods / len(all_methods), 1.0)
    
    def _calculate_domain_relevance(self, papers: List[Dict], query: str) -> float:
        """Calculate domain relevance"""
        if not papers:
            return 0.0
        
        # Simple keyword matching
        query_lower = query.lower()
        relevant_papers = 0
        
        for paper in papers:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            
            # Check if query keywords appear in title or abstract
            query_words = query_lower.split()
            title_abstract = f"{title} {abstract}"
            
            if any(word in title_abstract for word in query_words):
                relevant_papers += 1
        
        return relevant_papers / len(papers)
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _calculate_precision(self, predictions: List[bool], ground_truth: List[bool]) -> float:
        """Calculate precision"""
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
        
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _calculate_recall(self, predictions: List[bool], ground_truth: List[bool]) -> float:
        """Calculate recall"""
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
        
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

class ScientificBenchmarkSuite:
    """Complete benchmark suite for scientific research capabilities"""
    
    def __init__(self):
        self.evaluator = ScientificEvaluator()
        self.benchmark_results = {}
    
    async def run_complete_benchmark(self, scientific_system: Dict[str, Any], 
                                   dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete scientific research benchmark"""
        
        print("🧬 Starting Scientific Research Benchmark...")
        start_time = datetime.now()
        
        benchmark_results = {
            "benchmark_id": f"sci_bench_{int(start_time.timestamp())}",
            "started_at": start_time.isoformat(),
            "results": {}
        }
        
        # Test 1: Paper Retrieval
        print("📄 Testing Paper Retrieval...")
        retrieval_results = await self._test_paper_retrieval(scientific_system, dataset)
        benchmark_results["results"]["paper_retrieval"] = retrieval_results
        
        # Test 2: Novelty Detection
        print("🆕 Testing Novelty Detection...")
        novelty_results = await self._test_novelty_detection(scientific_system, dataset)
        benchmark_results["results"]["novelty_detection"] = novelty_results
        
        # Test 3: Impact Prediction
        print("📈 Testing Impact Prediction...")
        impact_results = await self._test_impact_prediction(scientific_system, dataset)
        benchmark_results["results"]["impact_prediction"] = impact_results
        
        # Test 4: Trend Analysis
        print("📊 Testing Trend Analysis...")
        trend_results = await self._test_trend_analysis(scientific_system, dataset)
        benchmark_results["results"]["trend_analysis"] = trend_results
        
        # Test 5: Research Gap Identification
        print("🔍 Testing Research Gap Identification...")
        gap_results = await self._test_gap_identification(scientific_system, dataset)
        benchmark_results["results"]["gap_identification"] = gap_results
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(benchmark_results["results"])
        benchmark_results["overall_score"] = overall_score
        
        end_time = datetime.now()
        benchmark_results["completed_at"] = end_time.isoformat()
        benchmark_results["duration"] = (end_time - start_time).total_seconds()
        
        print(f"✅ Scientific Research Benchmark Completed!")
        print(f"Overall Score: {overall_score:.3f}")
        
        return benchmark_results
    
    async def _test_paper_retrieval(self, system: Dict[str, Any], 
                                   dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Test paper retrieval capabilities"""
        
        queries = dataset.get("queries", [])
        qrels = dataset.get("qrels", [])
        
        if not queries:
            return {"error": "No queries in dataset"}
        
        all_metrics = []
        
        for query in queries[:3]:  # Test first 3 queries
            query_text = query.get("text", "")
            query_id = query.get("query_id", "")
            
            # Retrieve papers using scientific system
            if "retriever" in system:
                retrieved_papers = system["retriever"].retrieve_scientific_papers(query_text)
            else:
                # Mock retrieval
                retrieved_papers = dataset.get("papers", [])[:5]
            
            # Get ground truth
            ground_truth = [qrel for qrel in qrels if qrel.get("query_id") == query_id]
            
            # Evaluate
            metrics = self.evaluator.evaluate_paper_retrieval(query_text, retrieved_papers, ground_truth)
            all_metrics.append(metrics)
        
        # Average metrics
        if all_metrics:
            avg_metrics = {}
            for field in ["precision", "recall", "f1", "novelty_coverage", "impact_coverage"]:
                values = [getattr(m, field) for m in all_metrics]
                avg_metrics[field] = sum(values) / len(values)
            
            return avg_metrics
        
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    async def _test_novelty_detection(self, system: Dict[str, Any], 
                                     dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Test novelty detection capabilities"""
        
        papers = dataset.get("papers", [])
        if not papers:
            return {"error": "No papers in dataset"}
        
        # Generate novelty predictions
        novelty_predictions = []
        
        for paper in papers:
            if "research_agent" in system:
                analysis = system["research_agent"].analyze_paper(paper)
                novelty_score = analysis.get("novelty_analysis", {}).get("novelty_score", 0.5)
            else:
                # Mock prediction based on year
                year = paper.get("year", 2020)
                novelty_score = min((year - 2015) / 10.0, 1.0)
            
            novelty_predictions.append(novelty_score)
        
        # Evaluate
        return self.evaluator.evaluate_novelty_detection(papers, novelty_predictions)
    
    async def _test_impact_prediction(self, system: Dict[str, Any], 
                                     dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Test impact prediction capabilities"""
        
        papers = dataset.get("papers", [])
        if not papers:
            return {"error": "No papers in dataset"}
        
        # Generate impact predictions
        impact_predictions = []
        
        for paper in papers:
            if "research_agent" in system:
                analysis = system["research_agent"].analyze_paper(paper)
                impact_score = analysis.get("impact_analysis", {}).get("impact_score", 0.5)
            else:
                # Mock prediction based on venue and authors
                venue = paper.get("venue", "").lower()
                author_count = len(paper.get("authors", []))
                
                venue_score = 0.9 if venue in ["neurips", "icml", "nature"] else 0.5
                collab_score = min(author_count / 10.0, 0.3)
                impact_score = venue_score + collab_score
            
            impact_predictions.append(min(impact_score, 1.0))
        
        # Evaluate
        return self.evaluator.evaluate_impact_prediction(papers, impact_predictions)
    
    async def _test_trend_analysis(self, system: Dict[str, Any], 
                                  dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Test trend analysis capabilities"""
        
        if "research_agent" in system:
            trend_results = system["research_agent"].generate_research_insights()
        else:
            # Mock trend results
            trend_results = {
                "top_methods": [("transformer", 10), ("bert", 8), ("gpt", 6)],
                "trends": {
                    "emerging_methods": [{"method": "diffusion", "growth_ratio": 3.0}],
                    "year_distribution": {2020: 5, 2021: 8, 2022: 12}
                }
            }
        
        return self.evaluator.evaluate_trend_analysis(trend_results)
    
    async def _test_gap_identification(self, system: Dict[str, Any], 
                                      dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Test research gap identification"""
        
        if "research_agent" in system:
            # Generate research insights which include gaps
            insights = system["research_agent"].generate_research_insights()
            identified_gaps = insights.get("research_gaps", [])
        else:
            # Mock gap identification
            identified_gaps = [
                "Limited multilingual evaluation",
                "Insufficient human evaluation studies",
                "Limited reproducibility studies"
            ]
        
        return self.evaluator.evaluate_research_gaps(identified_gaps)
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall benchmark score"""
        
        scores = []
        weights = {
            "paper_retrieval": 0.3,
            "novelty_detection": 0.2,
            "impact_prediction": 0.2,
            "trend_analysis": 0.15,
            "gap_identification": 0.15
        }
        
        for task, weight in weights.items():
            if task in results:
                task_results = results[task]
                
                # Extract primary metric for each task
                if task == "paper_retrieval":
                    score = task_results.get("f1", 0.0)
                elif task == "novelty_detection":
                    score = task_results.get("novelty_f1", 0.0)
                elif task == "impact_prediction":
                    score = task_results.get("impact_correlation", 0.0)
                elif task == "trend_analysis":
                    score = task_results.get("trend_accuracy", 0.0)
                elif task == "gap_identification":
                    score = task_results.get("gap_detection_f1", 0.0)
                else:
                    score = 0.0
                
                scores.append(score * weight)
        
        return sum(scores)

# Factory functions
def create_scientific_evaluator() -> ScientificEvaluator:
    """Create scientific evaluator"""
    return ScientificEvaluator()

def create_scientific_benchmark_suite() -> ScientificBenchmarkSuite:
    """Create scientific benchmark suite"""
    return ScientificBenchmarkSuite()