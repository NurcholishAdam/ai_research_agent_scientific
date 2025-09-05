# -*- coding: utf-8 -*-
"""
Scientific Research Core Extensions
Core scientific research capabilities for AI Research Agent
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class ScientificPaper:
    """Represents a scientific paper with research metadata"""
    title: str
    authors: List[str]
    venue: str
    year: int
    abstract: str = ""
    doi: str = ""
    arxiv_id: str = ""
    citation_count: int = 0
    research_domain: str = "computer_science"
    methodology: List[str] = None
    datasets: List[str] = None
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.methodology is None:
            self.methodology = []
        if self.datasets is None:
            self.datasets = []
        if self.metrics is None:
            self.metrics = []

@dataclass
class ResearchConcept:
    """Represents a scientific concept or method"""
    name: str
    definition: str
    domain: str
    aliases: List[str] = None
    related_concepts: List[str] = None
    first_paper: str = ""
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.related_concepts is None:
            self.related_concepts = []

class ScientificEntityExtractor:
    """Extracts scientific entities from research text"""
    
    def __init__(self):
        # Common scientific patterns
        self.method_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:algorithm|method|approach)\b',
            r'\b(?:deep|neural|machine)\s+learning\b',
            r'\btransformer\s+(?:model|architecture)\b',
            r'\b(?:BERT|GPT|T5|RoBERTa|ELECTRA)\b',
            r'\bconvolutional\s+neural\s+network\b',
            r'\brecurrent\s+neural\s+network\b'
        ]
        
        self.metric_patterns = [
            r'\b(?:accuracy|precision|recall|F1|BLEU|ROUGE)\b',
            r'\b(?:perplexity|loss|error\s+rate)\b',
            r'\b(?:AUC|ROC|mAP|IoU)\b'
        ]
        
        self.dataset_patterns = [
            r'\b(?:ImageNet|COCO|MNIST|CIFAR)\b',
            r'\b(?:SQuAD|GLUE|SuperGLUE)\b',
            r'\b(?:WMT|CoNLL|ACE)\b'
        ]
    
    def extract_methods(self, text: str) -> List[str]:
        """Extract methodology mentions from text"""
        methods = []
        for pattern in self.method_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            methods.extend(matches)
        return list(set(methods))
    
    def extract_metrics(self, text: str) -> List[str]:
        """Extract evaluation metrics from text"""
        metrics = []
        for pattern in self.metric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            metrics.extend(matches)
        return list(set(metrics))
    
    def extract_datasets(self, text: str) -> List[str]:
        """Extract dataset mentions from text"""
        datasets = []
        for pattern in self.dataset_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            datasets.extend(matches)
        return list(set(datasets))

class CitationAnalyzer:
    """Analyzes citation patterns and relationships"""
    
    def __init__(self):
        self.citation_contexts = {
            'background': ['background', 'prior work', 'previous studies'],
            'methodology': ['method', 'approach', 'technique', 'algorithm'],
            'comparison': ['compared to', 'versus', 'outperforms', 'better than'],
            'extension': ['extends', 'builds on', 'based on', 'inspired by'],
            'criticism': ['however', 'limitation', 'problem with', 'fails to']
        }
    
    def analyze_citation_context(self, text: str, citation: str) -> str:
        """Determine the context in which a citation appears"""
        text_lower = text.lower()
        citation_lower = citation.lower()
        
        # Find citation position
        citation_pos = text_lower.find(citation_lower)
        if citation_pos == -1:
            return "unknown"
        
        # Analyze surrounding text
        window_size = 100
        start = max(0, citation_pos - window_size)
        end = min(len(text), citation_pos + len(citation) + window_size)
        context_text = text_lower[start:end]
        
        # Check for context indicators
        for context_type, indicators in self.citation_contexts.items():
            for indicator in indicators:
                if indicator in context_text:
                    return context_type
        
        return "general"

class ResearchTrendAnalyzer:
    """Analyzes research trends and patterns"""
    
    def __init__(self):
        self.trend_indicators = {
            'emerging': ['novel', 'new', 'recent', 'emerging', 'state-of-the-art'],
            'declining': ['traditional', 'classical', 'outdated', 'superseded'],
            'stable': ['established', 'standard', 'widely used', 'common']
        }
    
    def analyze_concept_trend(self, papers: List[ScientificPaper], concept: str) -> Dict[str, Any]:
        """Analyze trend for a specific concept"""
        concept_papers = [p for p in papers if concept.lower() in p.title.lower() or 
                         concept.lower() in p.abstract.lower()]
        
        if not concept_papers:
            return {"trend": "unknown", "papers": 0, "years": []}
        
        years = [p.year for p in concept_papers]
        recent_papers = [p for p in concept_papers if p.year >= 2020]
        
        trend_analysis = {
            "concept": concept,
            "total_papers": len(concept_papers),
            "recent_papers": len(recent_papers),
            "years": sorted(set(years)),
            "peak_year": max(set(years), key=years.count) if years else None,
            "trend": self._determine_trend(years),
            "citation_impact": sum(p.citation_count for p in concept_papers)
        }
        
        return trend_analysis
    
    def _determine_trend(self, years: List[int]) -> str:
        """Determine if concept is emerging, stable, or declining"""
        if not years:
            return "unknown"
        
        current_year = datetime.now().year
        recent_years = [y for y in years if y >= current_year - 3]
        older_years = [y for y in years if y < current_year - 3]
        
        if len(recent_years) > len(older_years):
            return "emerging"
        elif len(recent_years) < len(older_years) * 0.5:
            return "declining"
        else:
            return "stable"

class ScientificKnowledgeGraph:
    """Manages scientific knowledge relationships"""
    
    def __init__(self):
        self.concepts = {}
        self.papers = {}
        self.relationships = []
        self.entity_extractor = ScientificEntityExtractor()
        self.citation_analyzer = CitationAnalyzer()
    
    def add_paper(self, paper: ScientificPaper) -> str:
        """Add a scientific paper to the knowledge graph"""
        paper_id = f"paper_{len(self.papers)}"
        
        # Extract scientific entities
        full_text = f"{paper.title} {paper.abstract}"
        paper.methodology = self.entity_extractor.extract_methods(full_text)
        paper.datasets = self.entity_extractor.extract_datasets(full_text)
        paper.metrics = self.entity_extractor.extract_metrics(full_text)
        
        self.papers[paper_id] = paper
        
        # Create concept nodes
        for method in paper.methodology:
            self._add_concept(method, "methodology", paper.research_domain)
        
        for dataset in paper.datasets:
            self._add_concept(dataset, "dataset", paper.research_domain)
        
        for metric in paper.metrics:
            self._add_concept(metric, "metric", paper.research_domain)
        
        return paper_id
    
    def _add_concept(self, name: str, concept_type: str, domain: str):
        """Add a concept to the knowledge graph"""
        concept_id = f"{concept_type}_{name.lower().replace(' ', '_')}"
        
        if concept_id not in self.concepts:
            concept = ResearchConcept(
                name=name,
                definition=f"A {concept_type} in {domain}",
                domain=domain
            )
            self.concepts[concept_id] = concept
    
    def find_related_papers(self, query_paper: ScientificPaper, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find papers related to the query paper"""
        similarities = []
        
        for paper_id, paper in self.papers.items():
            similarity = self._calculate_paper_similarity(query_paper, paper)
            similarities.append((paper_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _calculate_paper_similarity(self, paper1: ScientificPaper, paper2: ScientificPaper) -> float:
        """Calculate similarity between two papers"""
        # Simple similarity based on shared entities
        shared_methods = set(paper1.methodology) & set(paper2.methodology)
        shared_datasets = set(paper1.datasets) & set(paper2.datasets)
        shared_metrics = set(paper1.metrics) & set(paper2.metrics)
        
        total_shared = len(shared_methods) + len(shared_datasets) + len(shared_metrics)
        total_entities = len(set(paper1.methodology + paper1.datasets + paper1.metrics + 
                                paper2.methodology + paper2.datasets + paper2.metrics))
        
        return total_shared / max(total_entities, 1)
    
    def get_concept_evolution(self, concept_name: str) -> Dict[str, Any]:
        """Get evolution of a concept over time"""
        related_papers = []
        
        for paper in self.papers.values():
            if (concept_name.lower() in [m.lower() for m in paper.methodology] or
                concept_name.lower() in paper.title.lower() or
                concept_name.lower() in paper.abstract.lower()):
                related_papers.append(paper)
        
        if not related_papers:
            return {"concept": concept_name, "evolution": "No data available"}
        
        # Sort by year
        related_papers.sort(key=lambda p: p.year)
        
        evolution = {
            "concept": concept_name,
            "first_appearance": related_papers[0].year,
            "latest_appearance": related_papers[-1].year,
            "total_papers": len(related_papers),
            "yearly_distribution": self._get_yearly_distribution(related_papers),
            "key_papers": [{"title": p.title, "year": p.year, "citations": p.citation_count} 
                          for p in related_papers[:5]]
        }
        
        return evolution
    
    def _get_yearly_distribution(self, papers: List[ScientificPaper]) -> Dict[int, int]:
        """Get yearly distribution of papers"""
        distribution = {}
        for paper in papers:
            distribution[paper.year] = distribution.get(paper.year, 0) + 1
        return distribution

class ScientificResearchAgent:
    """Main agent for scientific research tasks"""
    
    def __init__(self):
        self.knowledge_graph = ScientificKnowledgeGraph()
        self.trend_analyzer = ResearchTrendAnalyzer()
        self.research_history = []
    
    def analyze_research_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis of a research paper"""
        # Create paper object
        paper = ScientificPaper(
            title=paper_data.get("title", ""),
            authors=paper_data.get("authors", []),
            venue=paper_data.get("venue", ""),
            year=paper_data.get("year", 2024),
            abstract=paper_data.get("abstract", ""),
            doi=paper_data.get("doi", ""),
            citation_count=paper_data.get("citations", 0)
        )
        
        # Add to knowledge graph
        paper_id = self.knowledge_graph.add_paper(paper)
        
        # Find related papers
        related_papers = self.knowledge_graph.find_related_papers(paper)
        
        # Analyze novelty and impact
        analysis = {
            "paper_id": paper_id,
            "extracted_methods": paper.methodology,
            "extracted_datasets": paper.datasets,
            "extracted_metrics": paper.metrics,
            "related_papers": related_papers,
            "research_domain": paper.research_domain,
            "novelty_indicators": self._assess_novelty(paper),
            "impact_prediction": self._predict_impact(paper),
            "research_gaps": self._identify_gaps(paper)
        }
        
        return analysis
    
    def _assess_novelty(self, paper: ScientificPaper) -> Dict[str, Any]:
        """Assess the novelty of a research paper"""
        novelty_score = 0.5  # Base score
        
        # Check for novel methods
        novel_methods = []
        for method in paper.methodology:
            if method not in [c.name for c in self.knowledge_graph.concepts.values()]:
                novel_methods.append(method)
                novelty_score += 0.1
        
        # Check for novel combinations
        method_combinations = len(paper.methodology) > 1
        if method_combinations:
            novelty_score += 0.1
        
        return {
            "novelty_score": min(novelty_score, 1.0),
            "novel_methods": novel_methods,
            "novel_combinations": method_combinations,
            "assessment": "high" if novelty_score > 0.7 else "medium" if novelty_score > 0.4 else "low"
        }
    
    def _predict_impact(self, paper: ScientificPaper) -> Dict[str, Any]:
        """Predict potential impact of a research paper"""
        impact_score = 0.5  # Base score
        
        # Venue impact
        high_impact_venues = ["Nature", "Science", "ICML", "NeurIPS", "ICLR", "ACL", "EMNLP"]
        if paper.venue in high_impact_venues:
            impact_score += 0.2
        
        # Author reputation (simplified)
        if len(paper.authors) > 5:  # Large collaboration
            impact_score += 0.1
        
        # Methodology impact
        if len(paper.methodology) > 2:  # Multiple methods
            impact_score += 0.1
        
        # Dataset novelty
        if len(paper.datasets) > 1:  # Multiple datasets
            impact_score += 0.1
        
        return {
            "impact_score": min(impact_score, 1.0),
            "prediction": "high" if impact_score > 0.7 else "medium" if impact_score > 0.4 else "low",
            "factors": {
                "venue_prestige": paper.venue in high_impact_venues,
                "collaboration_size": len(paper.authors),
                "methodological_diversity": len(paper.methodology),
                "dataset_diversity": len(paper.datasets)
            }
        }
    
    def _identify_gaps(self, paper: ScientificPaper) -> List[str]:
        """Identify potential research gaps"""
        gaps = []
        
        # Method gaps
        if not paper.methodology:
            gaps.append("No clear methodology identified")
        
        # Evaluation gaps
        if not paper.metrics:
            gaps.append("Limited evaluation metrics")
        
        # Dataset gaps
        if not paper.datasets:
            gaps.append("No standard datasets used")
        
        # Reproducibility gaps
        if not paper.doi and not paper.arxiv_id:
            gaps.append("Limited accessibility for reproduction")
        
        return gaps
    
    def generate_research_insights(self, domain: str = "computer_science") -> Dict[str, Any]:
        """Generate insights about research trends in a domain"""
        domain_papers = [p for p in self.knowledge_graph.papers.values() 
                        if p.research_domain == domain]
        
        if not domain_papers:
            return {"domain": domain, "insights": "No papers available for analysis"}
        
        # Analyze trends
        all_methods = []
        for paper in domain_papers:
            all_methods.extend(paper.methodology)
        
        method_frequency = {}
        for method in all_methods:
            method_frequency[method] = method_frequency.get(method, 0) + 1
        
        # Top methods
        top_methods = sorted(method_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        insights = {
            "domain": domain,
            "total_papers": len(domain_papers),
            "top_methods": top_methods,
            "recent_trends": self._analyze_recent_trends(domain_papers),
            "emerging_areas": self._identify_emerging_areas(domain_papers),
            "collaboration_patterns": self._analyze_collaborations(domain_papers)
        }
        
        return insights
    
    def _analyze_recent_trends(self, papers: List[ScientificPaper]) -> Dict[str, Any]:
        """Analyze recent trends in papers"""
        current_year = datetime.now().year
        recent_papers = [p for p in papers if p.year >= current_year - 2]
        
        if not recent_papers:
            return {"trend": "No recent papers"}
        
        recent_methods = []
        for paper in recent_papers:
            recent_methods.extend(paper.methodology)
        
        method_counts = {}
        for method in recent_methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        trending_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "recent_papers_count": len(recent_papers),
            "trending_methods": trending_methods,
            "average_citations": sum(p.citation_count for p in recent_papers) / len(recent_papers)
        }
    
    def _identify_emerging_areas(self, papers: List[ScientificPaper]) -> List[str]:
        """Identify emerging research areas"""
        # Simple heuristic: methods that appear frequently in recent papers
        current_year = datetime.now().year
        recent_papers = [p for p in papers if p.year >= current_year - 1]
        older_papers = [p for p in papers if p.year < current_year - 1]
        
        recent_methods = set()
        for paper in recent_papers:
            recent_methods.update(paper.methodology)
        
        older_methods = set()
        for paper in older_papers:
            older_methods.update(paper.methodology)
        
        emerging = list(recent_methods - older_methods)
        return emerging[:5]  # Top 5 emerging areas
    
    def _analyze_collaborations(self, papers: List[ScientificPaper]) -> Dict[str, Any]:
        """Analyze collaboration patterns"""
        author_counts = [len(p.authors) for p in papers]
        
        if not author_counts:
            return {"collaboration": "No data"}
        
        return {
            "average_authors": sum(author_counts) / len(author_counts),
            "max_collaboration": max(author_counts),
            "single_author_papers": sum(1 for count in author_counts if count == 1),
            "large_collaborations": sum(1 for count in author_counts if count > 5)
        }

# Factory function
def create_scientific_research_agent() -> ScientificResearchAgent:
    """Create a scientific research agent"""
    return ScientificResearchAgent()
