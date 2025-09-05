# -*- coding: utf-8 -*-
"""
Scientific LIMIT-GRAPH Extension
Extends LIMIT-GRAPH with scientific research capabilities
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from .limit_graph_core import LimitGraphConfig, LimitGraphScaffold, LIMIT_GRAPH_REGISTRY
from .scientific_research_core import ScientificResearchAgent, ScientificPaper, ResearchConcept

logger = logging.getLogger(__name__)

@dataclass
class ScientificLimitConfig(LimitGraphConfig):
    """Extended configuration for scientific LIMIT-GRAPH"""
    enable_citation_analysis: bool = True
    enable_trend_analysis: bool = True
    enable_novelty_detection: bool = True
    enable_impact_prediction: bool = True
    scientific_domains: List[str] = None
    min_citation_threshold: int = 5
    novelty_threshold: float = 0.7
    
    def __post_init__(self):
        super().__post_init__()
        if self.scientific_domains is None:
            self.scientific_domains = [
                "computer_science", "artificial_intelligence", 
                "machine_learning", "natural_language_processing"
            ]

class ScientificGraphScaffold(LimitGraphScaffold):
    """Scientific extension of graph scaffold"""
    
    def __init__(self, config: ScientificLimitConfig):
        super().__init__(config)
        self.scientific_config = config
        self.research_agent = ScientificResearchAgent()
        self.paper_registry = {}
        self.concept_registry = {}
        
    def process_scientific_corpus(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process scientific papers corpus"""
        
        processing_stats = {
            "papers_processed": 0,
            "concepts_extracted": 0,
            "citations_analyzed": 0,
            "novel_papers": 0,
            "high_impact_papers": 0
        }
        
        for paper_data in papers:
            # Analyze paper with scientific agent
            analysis = self.research_agent.analyze_research_paper(paper_data)
            
            # Create scientific paper object
            paper = ScientificPaper(
                title=paper_data.get("title", ""),
                authors=paper_data.get("authors", []),
                venue=paper_data.get("venue", ""),
                year=paper_data.get("year", 2024),
                abstract=paper_data.get("abstract", ""),
                citation_count=paper_data.get("citations", 0)
            )
            
            paper_id = f"paper_{processing_stats['papers_processed']}"
            self.paper_registry[paper_id] = paper
            
            # Add to graph
            self._add_paper_to_graph(paper_id, paper, analysis)
            
            # Update statistics
            processing_stats["papers_processed"] += 1
            processing_stats["concepts_extracted"] += len(analysis["extracted_methods"])
            
            if analysis["novelty_indicators"]["novelty_score"] > self.scientific_config.novelty_threshold:
                processing_stats["novel_papers"] += 1
            
            if analysis["impact_prediction"]["impact_score"] > 0.7:
                processing_stats["high_impact_papers"] += 1
        
        # Generate scientific insights
        insights = self.research_agent.generate_research_insights()
        processing_stats["research_insights"] = insights
        
        return processing_stats
    
    def _add_paper_to_graph(self, paper_id: str, paper: ScientificPaper, analysis: Dict[str, Any]):
        """Add scientific paper to graph with rich metadata"""
        
        # Add paper node
        self.graph_nodes[paper_id] = {
            "type": "scientific_paper",
            "title": paper.title,
            "authors": paper.authors,
            "venue": paper.venue,
            "year": paper.year,
            "citation_count": paper.citation_count,
            "methodology": analysis["extracted_methods"],
            "datasets": analysis["extracted_datasets"],
            "metrics": analysis["extracted_metrics"],
            "novelty_score": analysis["novelty_indicators"]["novelty_score"],
            "impact_score": analysis["impact_prediction"]["impact_score"],
            "research_gaps": analysis["research_gaps"]
        }
        
        # Add method nodes and edges
        for method in analysis["extracted_methods"]:
            method_id = f"method_{method.lower().replace(' ', '_')}"
            
            if method_id not in self.graph_nodes:
                self.graph_nodes[method_id] = {
                    "type": "methodology",
                    "name": method,
                    "papers": [],
                    "domain": "computer_science"
                }
            
            self.graph_nodes[method_id]["papers"].append(paper_id)
            
            # Add uses_method edge
            self.graph_edges.append({
                "source": paper_id,
                "target": method_id,
                "relation": "uses_method",
                "confidence": 1.0,
                "context": "methodology"
            })
        
        # Add dataset nodes and edges
        for dataset in analysis["extracted_datasets"]:
            dataset_id = f"dataset_{dataset.lower().replace(' ', '_')}"
            
            if dataset_id not in self.graph_nodes:
                self.graph_nodes[dataset_id] = {
                    "type": "dataset",
                    "name": dataset,
                    "papers": [],
                    "domain": "computer_science"
                }
            
            self.graph_nodes[dataset_id]["papers"].append(paper_id)
            
            # Add uses_dataset edge
            self.graph_edges.append({
                "source": paper_id,
                "target": dataset_id,
                "relation": "uses_dataset",
                "confidence": 1.0,
                "context": "evaluation"
            })
        
        # Add metric nodes and edges
        for metric in analysis["extracted_metrics"]:
            metric_id = f"metric_{metric.lower().replace(' ', '_')}"
            
            if metric_id not in self.graph_nodes:
                self.graph_nodes[metric_id] = {
                    "type": "metric",
                    "name": metric,
                    "papers": [],
                    "domain": "computer_science"
                }
            
            self.graph_nodes[metric_id]["papers"].append(paper_id)
            
            # Add reports_metric edge
            self.graph_edges.append({
                "source": paper_id,
                "target": metric_id,
                "relation": "reports_metric",
                "confidence": 1.0,
                "context": "evaluation"
            })

class ScientificHybridRetriever:
    """Scientific-aware hybrid retriever"""
    
    def __init__(self, graph_scaffold: ScientificGraphScaffold, config: ScientificLimitConfig):
        self.graph_scaffold = graph_scaffold
        self.config = config
    
    def retrieve_scientific_papers(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve scientific papers with domain-specific ranking"""
        
        if filters is None:
            filters = {}
        
        # Extract scientific entities from query
        query_methods = self.graph_scaffold.research_agent.knowledge_graph.entity_extractor.extract_methods(query)
        query_datasets = self.graph_scaffold.research_agent.knowledge_graph.entity_extractor.extract_datasets(query)
        query_metrics = self.graph_scaffold.research_agent.knowledge_graph.entity_extractor.extract_metrics(query)
        
        results = []
        
        for paper_id, paper_data in self.graph_scaffold.graph_nodes.items():
            if paper_data.get("type") != "scientific_paper":
                continue
            
            # Apply filters
            if not self._passes_filters(paper_data, filters):
                continue
            
            # Calculate scientific relevance score
            relevance_score = self._calculate_scientific_relevance(
                paper_data, query, query_methods, query_datasets, query_metrics
            )
            
            if relevance_score > 0:
                results.append({
                    "paper_id": paper_id,
                    "title": paper_data["title"],
                    "authors": paper_data["authors"],
                    "venue": paper_data["venue"],
                    "year": paper_data["year"],
                    "relevance_score": relevance_score,
                    "novelty_score": paper_data.get("novelty_score", 0.5),
                    "impact_score": paper_data.get("impact_score", 0.5),
                    "citation_count": paper_data.get("citation_count", 0),
                    "methodology": paper_data.get("methodology", []),
                    "research_gaps": paper_data.get("research_gaps", [])
                })
        
        # Sort by combined score (relevance + impact + novelty)
        results.sort(key=lambda x: (
            x["relevance_score"] * 0.4 + 
            x["impact_score"] * 0.3 + 
            x["novelty_score"] * 0.3
        ), reverse=True)
        
        return results[:self.config.retrieval_top_k]
    
    def _passes_filters(self, paper_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if paper passes the specified filters"""
        
        # Year filter
        if "min_year" in filters and paper_data.get("year", 0) < filters["min_year"]:
            return False
        
        if "max_year" in filters and paper_data.get("year", 9999) > filters["max_year"]:
            return False
        
        # Citation filter
        if "min_citations" in filters and paper_data.get("citation_count", 0) < filters["min_citations"]:
            return False
        
        # Venue filter
        if "venues" in filters and paper_data.get("venue", "") not in filters["venues"]:
            return False
        
        # Methodology filter
        if "required_methods" in filters:
            paper_methods = set(paper_data.get("methodology", []))
            required_methods = set(filters["required_methods"])
            if not required_methods.intersection(paper_methods):
                return False
        
        return True
    
    def _calculate_scientific_relevance(self, paper_data: Dict[str, Any], query: str,
                                      query_methods: List[str], query_datasets: List[str],
                                      query_metrics: List[str]) -> float:
        """Calculate scientific relevance score"""
        
        score = 0.0
        
        # Text similarity (basic)
        query_lower = query.lower()
        title_lower = paper_data.get("title", "").lower()
        
        # Title match
        if any(word in title_lower for word in query_lower.split()):
            score += 0.3
        
        # Method overlap
        paper_methods = set(m.lower() for m in paper_data.get("methodology", []))
        query_methods_lower = set(m.lower() for m in query_methods)
        method_overlap = len(paper_methods.intersection(query_methods_lower))
        if method_overlap > 0:
            score += 0.4 * (method_overlap / max(len(query_methods_lower), 1))
        
        # Dataset overlap
        paper_datasets = set(d.lower() for d in paper_data.get("datasets", []))
        query_datasets_lower = set(d.lower() for d in query_datasets)
        dataset_overlap = len(paper_datasets.intersection(query_datasets_lower))
        if dataset_overlap > 0:
            score += 0.2 * (dataset_overlap / max(len(query_datasets_lower), 1))
        
        # Metric overlap
        paper_metrics = set(m.lower() for m in paper_data.get("metrics", []))
        query_metrics_lower = set(m.lower() for m in query_metrics)
        metric_overlap = len(paper_metrics.intersection(query_metrics_lower))
        if metric_overlap > 0:
            score += 0.1 * (metric_overlap / max(len(query_metrics_lower), 1))
        
        return min(score, 1.0)

class ScientificEvaluator:
    """Evaluator for scientific research tasks"""
    
    def __init__(self, config: ScientificLimitConfig):
        self.config = config
        self.evaluation_history = []
    
    def evaluate_scientific_retrieval(self, query: str, retrieved_papers: List[Dict[str, Any]],
                                    ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate scientific paper retrieval"""
        
        # Standard retrieval metrics
        retrieved_ids = {p["paper_id"] for p in retrieved_papers}
        relevant_ids = {p["paper_id"] for p in ground_truth}
        
        if not retrieved_ids or not relevant_ids:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        intersection = retrieved_ids.intersection(relevant_ids)
        
        precision = len(intersection) / len(retrieved_ids)
        recall = len(intersection) / len(relevant_ids)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Scientific-specific metrics
        scientific_metrics = self._calculate_scientific_metrics(retrieved_papers, ground_truth)
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **scientific_metrics
        }
        
        return metrics
    
    def _calculate_scientific_metrics(self, retrieved_papers: List[Dict[str, Any]],
                                    ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate scientific-specific evaluation metrics"""
        
        # Novelty coverage
        novel_retrieved = sum(1 for p in retrieved_papers if p.get("novelty_score", 0) > 0.7)
        novelty_coverage = novel_retrieved / len(retrieved_papers) if retrieved_papers else 0.0
        
        # Impact coverage
        high_impact_retrieved = sum(1 for p in retrieved_papers if p.get("impact_score", 0) > 0.7)
        impact_coverage = high_impact_retrieved / len(retrieved_papers) if retrieved_papers else 0.0
        
        # Citation diversity
        citation_counts = [p.get("citation_count", 0) for p in retrieved_papers]
        citation_diversity = len(set(citation_counts)) / len(citation_counts) if citation_counts else 0.0
        
        # Temporal diversity
        years = [p.get("year", 2024) for p in retrieved_papers]
        temporal_diversity = len(set(years)) / len(years) if years else 0.0
        
        # Methodological diversity
        all_methods = []
        for p in retrieved_papers:
            all_methods.extend(p.get("methodology", []))
        
        method_diversity = len(set(all_methods)) / len(all_methods) if all_methods else 0.0
        
        return {
            "novelty_coverage": novelty_coverage,
            "impact_coverage": impact_coverage,
            "citation_diversity": citation_diversity,
            "temporal_diversity": temporal_diversity,
            "method_diversity": method_diversity
        }

def create_scientific_limit_graph_system(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create scientific LIMIT-GRAPH system"""
    
    # Create configuration
    scientific_config = ScientificLimitConfig(**(config or {}))
    
    # Create components
    graph_scaffold = ScientificGraphScaffold(scientific_config)
    retriever = ScientificHybridRetriever(graph_scaffold, scientific_config)
    evaluator = ScientificEvaluator(scientific_config)
    
    # Register in LIMIT-GRAPH registry
    LIMIT_GRAPH_REGISTRY.register_component("scientific_graph_scaffold", graph_scaffold)
    LIMIT_GRAPH_REGISTRY.register_component("scientific_retriever", retriever)
    LIMIT_GRAPH_REGISTRY.register_component("scientific_evaluator", evaluator)
    
    system = {
        "config": scientific_config,
        "graph_scaffold": graph_scaffold,
        "retriever": retriever,
        "evaluator": evaluator,
        "research_agent": graph_scaffold.research_agent
    }
    
    logger.info("Scientific LIMIT-GRAPH system created successfully")
    
    return system

def create_sample_scientific_dataset() -> Dict[str, Any]:
    """Create sample scientific dataset"""
    
    sample_papers = [
        {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani", "Shazeer", "Parmar", "Uszkoreit"],
            "venue": "NeurIPS",
            "year": 2017,
            "abstract": "We propose the Transformer, a novel neural network architecture based solely on attention mechanisms.",
            "citations": 50000,
            "doi": "10.5555/3295222.3295349"
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Devlin", "Chang", "Lee", "Toutanova"],
            "venue": "NAACL",
            "year": 2019,
            "abstract": "We introduce BERT, a new language representation model based on bidirectional transformers.",
            "citations": 40000,
            "arxiv_id": "1810.04805"
        },
        {
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "authors": ["Brown", "Mann", "Ryder", "Subbiah"],
            "venue": "NeurIPS",
            "year": 2020,
            "abstract": "We show that scaling up language models greatly improves task-agnostic, few-shot performance.",
            "citations": 25000,
            "arxiv_id": "2005.14165"
        }
    ]
    
    sample_queries = [
        {
            "query_id": "sci_q1",
            "text": "What are the latest transformer architectures for natural language processing?",
            "domain": "natural_language_processing",
            "expected_methods": ["transformer", "attention", "BERT", "GPT"]
        },
        {
            "query_id": "sci_q2", 
            "text": "How do attention mechanisms work in neural networks?",
            "domain": "machine_learning",
            "expected_methods": ["attention", "neural network", "transformer"]
        }
    ]
    
    sample_qrels = [
        {
            "query_id": "sci_q1",
            "paper_id": "paper_0",
            "relevance": 2,
            "novelty": 1,
            "impact": 2
        },
        {
            "query_id": "sci_q1",
            "paper_id": "paper_1", 
            "relevance": 2,
            "novelty": 1,
            "impact": 2
        }
    ]
    
    return {
        "papers": sample_papers,
        "queries": sample_queries,
        "qrels": sample_qrels,
        "metadata": {
            "domain": "computer_science",
            "focus": "natural_language_processing",
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }
    }

# Register sample scientific dataset
scientific_dataset = create_sample_scientific_dataset()
LIMIT_GRAPH_REGISTRY.register_dataset("scientific_sample", scientific_dataset)