# -*- coding: utf-8 -*-
"""
Scientific Research Extensions for AI Research Agent
Advanced capabilities for academic and scientific research contexts
"""

import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ResearchDomain(Enum):
    """Scientific research domains"""
    COMPUTER_SCIENCE = "computer_science"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    MACHINE_LEARNING = "machine_learning"
    NATURAL_LANGUAGE_PROCESSING = "natural_language_processing"
    COGNITIVE_SCIENCE = "cognitive_science"
    NEUROSCIENCE = "neuroscience"
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MEDICINE = "medicine"
    PSYCHOLOGY = "psychology"
    INTERDISCIPLINARY = "interdisciplinary"

class PublicationType(Enum):
    """Types of scientific publications"""
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    WORKSHOP_PAPER = "workshop_paper"
    PREPRINT = "preprint"
    THESIS = "thesis"
    BOOK_CHAPTER = "book_chapter"
    TECHNICAL_REPORT = "technical_report"
    PATENT = "patent"
    DATASET = "dataset"
    SOFTWARE = "software"

class CitationContext(Enum):
    """Context in which a citation appears"""
    BACKGROUND = "background"
    METHODOLOGY = "methodology"
    COMPARISON = "comparison"
    EXTENSION = "extension"
    CRITICISM = "criticism"
    FUTURE_WORK = "future_work"
    RELATED_WORK = "related_work"

@dataclass
class ScientificEntity:
    """Represents a scientific entity (concept, method, dataset, etc.)"""
    name: str
    entity_type: str  # concept, method, dataset, metric, model, etc.
    domain: ResearchDomain
    aliases: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    mathematical_notation: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
    first_introduced: Optional[str] = None  # Paper/year where first introduced
    confidence: float = 1.0

@dataclass
class ResearchPaper:
    """Comprehensive representation of a research paper"""
    title: str
    authors: List[str]
    venue: str
    year: int
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    domains: List[ResearchDomain] = field(default_factory=list)
    publication_type: PublicationType = PublicationType.JOURNAL_ARTICLE
    citation_count: int = 0
    h_index_contribution: float = 0.0
    impact_factor: Optional[float] = None
    methodology: List[str] = field(default_factory=list)
    datasets_used: List[str] = field(default_factory=list)
    metrics_reported: List[str] = field(default_factory=list)
    reproducibility_score: Optional[float] = None
    novelty_score: Optional[float] = None

@dataclass
class ResearchHypothesis:
    """Represents a research hypothesis"""
    hypothesis_id: str
    statement: str
    domain: ResearchDomain
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    confidence_level: float = 0.5
    testability_score: float = 0.5
    related_hypotheses: List[str] = field(default_factory=list)
    experimental_design: Optional[str] = None

class ScientificEntityExtractor:
    """Extracts scientific entities from research text"""
    
    def __init__(self):
        # Scientific terminology patterns
        self.method_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:algorithm|method|approach|technique|model)\b',
            r'\b(?:algorithm|method|approach|technique|model)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b(transformer|BERT|GPT|attention|neural network|CNN|RNN|LSTM|GAN)\b',
            r'\b(deep learning|machine learning|reinforcement learning|supervised learning)\b',
            r'\b(classification|regression|clustering|optimization|fine-tuning)\b'
        ]
        
        self.dataset_patterns = [
            r'\b(ImageNet|COCO|MNIST|CIFAR|SQuAD|GLUE|SuperGLUE|WMT|CoNLL)\b',
            r'\b(Penn Treebank|Common Crawl|OpenWebText|BookCorpus)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:dataset|corpus|benchmark)\b'
        ]
        
        self.metric_patterns = [
            r'\b(accuracy|precision|recall|F1|BLEU|ROUGE|perplexity|loss)\b',
            r'\b(AUC|ROC|mAP|IoU|NDCG|MRR|MAP)\b',
            r'\b(error rate|success rate|coverage|diversity)\b'
        ]
        
        self.venue_patterns = [
            r'\b(NeurIPS|ICML|ICLR|ACL|EMNLP|NAACL|CVPR|ICCV|ECCV|AAAI|IJCAI)\b',
            r'\b(Nature|Science|Cell|PNAS|JMLR|TACL|CL|AI)\b'
        ]
    
    def extract_methods(self, text: str) -> List[ScientificEntity]:
        """Extract methodology entities from text"""
        methods = []
        for pattern in self.method_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                entity = ScientificEntity(
                    name=match,
                    entity_type="method",
                    domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
                    confidence=0.8
                )
                methods.append(entity)
        
        return methods
    
    def extract_datasets(self, text: str) -> List[ScientificEntity]:
        """Extract dataset entities from text"""
        datasets = []
        for pattern in self.dataset_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                entity = ScientificEntity(
                    name=match,
                    entity_type="dataset",
                    domain=ResearchDomain.COMPUTER_SCIENCE,
                    confidence=0.9
                )
                datasets.append(entity)
        
        return datasets
    
    def extract_metrics(self, text: str) -> List[ScientificEntity]:
        """Extract evaluation metrics from text"""
        metrics = []
        for pattern in self.metric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = ScientificEntity(
                    name=match,
                    entity_type="metric",
                    domain=ResearchDomain.MACHINE_LEARNING,
                    confidence=0.85
                )
                metrics.append(entity)
        
        return metrics

class CitationNetworkAnalyzer:
    """Analyzes citation networks and relationships"""
    
    def __init__(self):
        self.citation_graph = {}
        self.paper_metadata = {}
    
    def add_paper(self, paper: ResearchPaper, citations: List[str] = None):
        """Add paper to citation network"""
        paper_id = f"{paper.title}_{paper.year}"
        self.paper_metadata[paper_id] = paper
        
        if citations:
            self.citation_graph[paper_id] = citations
    
    def calculate_impact_metrics(self, paper_id: str) -> Dict[str, float]:
        """Calculate various impact metrics for a paper"""
        if paper_id not in self.paper_metadata:
            return {}
        
        paper = self.paper_metadata[paper_id]
        
        # Basic citation count
        citation_count = paper.citation_count
        
        # H-index contribution (simplified)
        h_index = min(citation_count, len([p for p in self.paper_metadata.values() 
                                         if p.citation_count >= citation_count]))
        
        # Temporal impact (citations per year since publication)
        years_since_pub = datetime.now().year - paper.year
        temporal_impact = citation_count / max(years_since_pub, 1)
        
        # Venue impact factor (simplified)
        venue_impact = self._get_venue_impact_factor(paper.venue)
        
        return {
            "citation_count": citation_count,
            "h_index_contribution": h_index,
            "temporal_impact": temporal_impact,
            "venue_impact_factor": venue_impact,
            "composite_impact": (citation_count * 0.4 + h_index * 0.3 + 
                               temporal_impact * 0.2 + venue_impact * 0.1)
        }
    
    def _get_venue_impact_factor(self, venue: str) -> float:
        """Get impact factor for venue"""
        venue_impacts = {
            "Nature": 1.0, "Science": 1.0, "Cell": 0.95,
            "NeurIPS": 0.9, "ICML": 0.9, "ICLR": 0.85,
            "ACL": 0.8, "EMNLP": 0.8, "NAACL": 0.75,
            "CVPR": 0.85, "ICCV": 0.85, "ECCV": 0.8,
            "AAAI": 0.7, "IJCAI": 0.7
        }
        return venue_impacts.get(venue, 0.5)
    
    def find_influential_papers(self, domain: ResearchDomain = None, 
                               top_k: int = 10) -> List[Tuple[str, Dict[str, float]]]:
        """Find most influential papers in domain"""
        paper_impacts = []
        
        for paper_id, paper in self.paper_metadata.items():
            if domain and domain not in paper.domains:
                continue
            
            impact_metrics = self.calculate_impact_metrics(paper_id)
            paper_impacts.append((paper_id, impact_metrics))
        
        # Sort by composite impact
        paper_impacts.sort(key=lambda x: x[1].get("composite_impact", 0), reverse=True)
        
        return paper_impacts[:top_k]

class ResearchTrendAnalyzer:
    """Analyzes research trends and evolution"""
    
    def __init__(self):
        self.papers = []
        self.temporal_analysis = {}
        self.domain_analysis = {}
    
    def add_papers(self, papers: List[ResearchPaper]):
        """Add papers for trend analysis"""
        self.papers.extend(papers)
        self._update_analyses()
    
    def _update_analyses(self):
        """Update temporal and domain analyses"""
        # Temporal analysis
        for paper in self.papers:
            year = paper.year
            if year not in self.temporal_analysis:
                self.temporal_analysis[year] = {
                    "paper_count": 0,
                    "methods": set(),
                    "datasets": set(),
                    "venues": set()
                }
            
            self.temporal_analysis[year]["paper_count"] += 1
            self.temporal_analysis[year]["methods"].update(paper.methodology)
            self.temporal_analysis[year]["datasets"].update(paper.datasets_used)
            self.temporal_analysis[year]["venues"].add(paper.venue)
        
        # Domain analysis
        for paper in self.papers:
            for domain in paper.domains:
                if domain not in self.domain_analysis:
                    self.domain_analysis[domain] = {
                        "paper_count": 0,
                        "avg_citations": 0,
                        "top_methods": {},
                        "top_venues": {}
                    }
                
                self.domain_analysis[domain]["paper_count"] += 1
                
                # Update method counts
                for method in paper.methodology:
                    if method not in self.domain_analysis[domain]["top_methods"]:
                        self.domain_analysis[domain]["top_methods"][method] = 0
                    self.domain_analysis[domain]["top_methods"][method] += 1
    
    def identify_emerging_trends(self, years_back: int = 3) -> Dict[str, Any]:
        """Identify emerging research trends"""
        current_year = datetime.now().year
        recent_years = range(current_year - years_back, current_year + 1)
        
        # Collect recent methods
        recent_methods = {}
        older_methods = {}
        
        for year, data in self.temporal_analysis.items():
            if year in recent_years:
                for method in data["methods"]:
                    recent_methods[method] = recent_methods.get(method, 0) + 1
            else:
                for method in data["methods"]:
                    older_methods[method] = older_methods.get(method, 0) + 1
        
        # Find emerging methods (high recent activity, low historical activity)
        emerging_methods = []
        for method, recent_count in recent_methods.items():
            older_count = older_methods.get(method, 0)
            if recent_count > older_count * 2:  # At least 2x growth
                emerging_methods.append({
                    "method": method,
                    "recent_count": recent_count,
                    "older_count": older_count,
                    "growth_ratio": recent_count / max(older_count, 1)
                })
        
        # Sort by growth ratio
        emerging_methods.sort(key=lambda x: x["growth_ratio"], reverse=True)
        
        return {
            "emerging_methods": emerging_methods[:10],
            "analysis_period": f"{current_year - years_back}-{current_year}",
            "total_recent_papers": sum(self.temporal_analysis.get(y, {}).get("paper_count", 0) 
                                     for y in recent_years)
        }
    
    def analyze_domain_evolution(self, domain: ResearchDomain) -> Dict[str, Any]:
        """Analyze evolution of a specific research domain"""
        domain_papers = [p for p in self.papers if domain in p.domains]
        
        if not domain_papers:
            return {"domain": domain.value, "analysis": "No papers found"}
        
        # Temporal evolution
        yearly_stats = {}
        for paper in domain_papers:
            year = paper.year
            if year not in yearly_stats:
                yearly_stats[year] = {
                    "papers": 0,
                    "total_citations": 0,
                    "methods": set(),
                    "venues": set()
                }
            
            yearly_stats[year]["papers"] += 1
            yearly_stats[year]["total_citations"] += paper.citation_count
            yearly_stats[year]["methods"].update(paper.methodology)
            yearly_stats[year]["venues"].add(paper.venue)
        
        # Calculate growth trends
        years = sorted(yearly_stats.keys())
        if len(years) > 1:
            recent_papers = yearly_stats[years[-1]]["papers"]
            older_papers = yearly_stats[years[0]]["papers"]
            growth_rate = (recent_papers - older_papers) / max(older_papers, 1)
        else:
            growth_rate = 0
        
        return {
            "domain": domain.value,
            "total_papers": len(domain_papers),
            "yearly_evolution": {year: {
                "papers": stats["papers"],
                "avg_citations": stats["total_citations"] / stats["papers"],
                "method_diversity": len(stats["methods"]),
                "venue_diversity": len(stats["venues"])
            } for year, stats in yearly_stats.items()},
            "growth_rate": growth_rate,
            "dominant_methods": self._get_dominant_methods(domain_papers),
            "key_venues": self._get_key_venues(domain_papers)
        }
    
    def _get_dominant_methods(self, papers: List[ResearchPaper]) -> List[Tuple[str, int]]:
        """Get dominant methods in paper set"""
        method_counts = {}
        for paper in papers:
            for method in paper.methodology:
                method_counts[method] = method_counts.get(method, 0) + 1
        
        return sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _get_key_venues(self, papers: List[ResearchPaper]) -> List[Tuple[str, int]]:
        """Get key venues in paper set"""
        venue_counts = {}
        for paper in papers:
            venue_counts[paper.venue] = venue_counts.get(paper.venue, 0) + 1
        
        return sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:10]

class ScientificResearchOrchestrator:
    """Main orchestrator for scientific research capabilities"""
    
    def __init__(self):
        self.entity_extractor = ScientificEntityExtractor()
        self.citation_analyzer = CitationNetworkAnalyzer()
        self.trend_analyzer = ResearchTrendAnalyzer()
        self.research_hypotheses = {}
        
    def analyze_research_corpus(self, papers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive analysis of research corpus"""
        
        # Convert to ResearchPaper objects
        papers = []
        for paper_data in papers_data:
            paper = ResearchPaper(
                title=paper_data.get("title", ""),
                authors=paper_data.get("authors", []),
                venue=paper_data.get("venue", ""),
                year=paper_data.get("year", 2024),
                abstract=paper_data.get("abstract", ""),
                doi=paper_data.get("doi"),
                citation_count=paper_data.get("citations", 0),
                domains=[ResearchDomain(d) for d in paper_data.get("domains", ["computer_science"])]
            )
            
            # Extract entities
            full_text = f"{paper.title} {paper.abstract}"
            methods = self.entity_extractor.extract_methods(full_text)
            datasets = self.entity_extractor.extract_datasets(full_text)
            metrics = self.entity_extractor.extract_metrics(full_text)
            
            paper.methodology = [m.name for m in methods]
            paper.datasets_used = [d.name for d in datasets]
            paper.metrics_reported = [m.name for m in metrics]
            
            papers.append(paper)
            
            # Add to citation network
            self.citation_analyzer.add_paper(paper)
        
        # Add to trend analyzer
        self.trend_analyzer.add_papers(papers)
        
        # Generate comprehensive analysis
        analysis = {
            "corpus_statistics": {
                "total_papers": len(papers),
                "year_range": f"{min(p.year for p in papers)}-{max(p.year for p in papers)}",
                "unique_venues": len(set(p.venue for p in papers)),
                "total_citations": sum(p.citation_count for p in papers),
                "avg_citations": sum(p.citation_count for p in papers) / len(papers)
            },
            "entity_analysis": self._analyze_extracted_entities(papers),
            "trend_analysis": self.trend_analyzer.identify_emerging_trends(),
            "domain_analysis": self._analyze_domains(papers),
            "impact_analysis": self._analyze_impact(papers),
            "research_gaps": self._identify_research_gaps(papers),
            "future_directions": self._suggest_future_directions(papers)
        }
        
        return analysis
    
    def _analyze_extracted_entities(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Analyze extracted entities across corpus"""
        all_methods = []
        all_datasets = []
        all_metrics = []
        
        for paper in papers:
            all_methods.extend(paper.methodology)
            all_datasets.extend(paper.datasets_used)
            all_metrics.extend(paper.metrics_reported)
        
        method_counts = {}
        for method in all_methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        dataset_counts = {}
        for dataset in all_datasets:
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        metric_counts = {}
        for metric in all_metrics:
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        return {
            "top_methods": sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:15],
            "top_datasets": sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_metrics": sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "method_diversity": len(set(all_methods)),
            "dataset_diversity": len(set(all_datasets)),
            "metric_diversity": len(set(all_metrics))
        }
    
    def _analyze_domains(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Analyze research domains"""
        domain_stats = {}
        
        for paper in papers:
            for domain in paper.domains:
                if domain not in domain_stats:
                    domain_stats[domain] = {
                        "paper_count": 0,
                        "total_citations": 0,
                        "methods": set(),
                        "venues": set()
                    }
                
                domain_stats[domain]["paper_count"] += 1
                domain_stats[domain]["total_citations"] += paper.citation_count
                domain_stats[domain]["methods"].update(paper.methodology)
                domain_stats[domain]["venues"].add(paper.venue)
        
        # Convert to serializable format
        domain_analysis = {}
        for domain, stats in domain_stats.items():
            domain_analysis[domain.value] = {
                "paper_count": stats["paper_count"],
                "avg_citations": stats["total_citations"] / stats["paper_count"],
                "method_diversity": len(stats["methods"]),
                "venue_diversity": len(stats["venues"]),
                "top_methods": list(stats["methods"])[:10]
            }
        
        return domain_analysis
    
    def _analyze_impact(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Analyze research impact patterns"""
        # High-impact papers (top 10% by citations)
        sorted_papers = sorted(papers, key=lambda p: p.citation_count, reverse=True)
        top_10_percent = int(len(papers) * 0.1)
        high_impact_papers = sorted_papers[:top_10_percent]
        
        # Analyze characteristics of high-impact papers
        high_impact_venues = {}
        high_impact_methods = {}
        
        for paper in high_impact_papers:
            high_impact_venues[paper.venue] = high_impact_venues.get(paper.venue, 0) + 1
            for method in paper.methodology:
                high_impact_methods[method] = high_impact_methods.get(method, 0) + 1
        
        return {
            "high_impact_threshold": high_impact_papers[-1].citation_count if high_impact_papers else 0,
            "high_impact_venues": sorted(high_impact_venues.items(), key=lambda x: x[1], reverse=True)[:10],
            "high_impact_methods": sorted(high_impact_methods.items(), key=lambda x: x[1], reverse=True)[:10],
            "citation_distribution": {
                "max": max(p.citation_count for p in papers),
                "min": min(p.citation_count for p in papers),
                "median": sorted([p.citation_count for p in papers])[len(papers)//2],
                "mean": sum(p.citation_count for p in papers) / len(papers)
            }
        }
    
    def _identify_research_gaps(self, papers: List[ResearchPaper]) -> List[str]:
        """Identify potential research gaps"""
        gaps = []
        
        # Method combination gaps
        all_methods = set()
        for paper in papers:
            all_methods.update(paper.methodology)
        
        common_methods = ["transformer", "bert", "gpt", "attention"]
        underexplored_methods = ["graph neural networks", "meta-learning", "few-shot learning"]
        
        for method in underexplored_methods:
            if method.lower() not in [m.lower() for m in all_methods]:
                gaps.append(f"Limited exploration of {method}")
        
        # Evaluation gaps
        all_metrics = set()
        for paper in papers:
            all_metrics.update(paper.metrics_reported)
        
        if "human evaluation" not in [m.lower() for m in all_metrics]:
            gaps.append("Limited human evaluation studies")
        
        # Domain gaps
        domain_counts = {}
        for paper in papers:
            for domain in paper.domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        if ResearchDomain.INTERDISCIPLINARY not in domain_counts:
            gaps.append("Limited interdisciplinary research")
        
        return gaps[:10]
    
    def _suggest_future_directions(self, papers: List[ResearchPaper]) -> List[str]:
        """Suggest future research directions"""
        directions = []
        
        # Based on trending methods
        trend_analysis = self.trend_analyzer.identify_emerging_trends()
        emerging_methods = trend_analysis.get("emerging_methods", [])
        
        if emerging_methods:
            top_emerging = emerging_methods[0]["method"]
            directions.append(f"Advanced applications of {top_emerging}")
            directions.append(f"Efficiency improvements for {top_emerging}")
        
        # Cross-domain opportunities
        directions.extend([
            "Cross-domain knowledge transfer",
            "Multimodal integration approaches",
            "Ethical AI and fairness considerations",
            "Sustainable and green AI methods",
            "Human-AI collaboration frameworks"
        ])
        
        return directions[:10]

# Factory function
def create_scientific_research_orchestrator() -> ScientificResearchOrchestrator:
    """Create scientific research orchestrator"""
    return ScientificResearchOrchestrator()