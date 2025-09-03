# tools/network_graph_analyzer.py
import json
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import hashlib

class NetworkGraphAnalyzer:
    """
    Analyzes investigation evidence to build and analyze network graphs of:
    - Accounts and their interactions
    - Media artifacts and their spread
    - Claims and their propagation
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()  # For tracking direction of information flow
        
    def load_evidence_from_file(self, filepath: str = "collected_evidence.json") -> Dict:
        """Load evidence from the collected evidence JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                evidence_data = json.load(f)
            return evidence_data
        except FileNotFoundError:
            print(f"Evidence file {filepath} not found")
            return {"evidence": []}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {"evidence": []}
    
    def build_network_from_evidence(self, evidence_data: Dict) -> Dict:
        """
        Build network graph from collected evidence
        
        Returns analysis including centrality measures and suspicious patterns
        """
        # Clear existing graph
        self.graph.clear()
        self.directed_graph.clear()
        
        evidence_list = evidence_data.get("evidence", [])
        
        # Track entities
        accounts = set()
        media_items = {}  # hash -> media info
        claims = set()
        websites = set()
        
        # Build nodes and edges from evidence
        for evidence in evidence_list:
            # Add account node
            author = evidence.get("author_id", "unknown")
            if author != "unknown":
                accounts.add(author)
                self._add_account_node(author, evidence)
            
            # Add media nodes and relationships
            if "media" in evidence:
                for media in evidence["media"]:
                    media_hash = self._get_media_hash(media["url"])
                    media_items[media_hash] = media
                    self._add_media_node(media_hash, media)
                    
                    # Add POSTED relationship
                    if author != "unknown":
                        self._add_edge(author, media_hash, "POSTED", evidence.get("timestamp"))
            
            # Add mentioned accounts and relationships
            mentioned = evidence.get("mentioned_accounts", [])
            for mentioned_account in mentioned:
                accounts.add(mentioned_account)
                self._add_account_node(mentioned_account, {})
                
                # Add MENTIONED relationship
                if author != "unknown":
                    self._add_edge(author, mentioned_account, "MENTIONED", evidence.get("timestamp"))
            
            # Add claims as nodes
            content = evidence.get("content", "")
            if content:
                claim_hash = hashlib.md5(content.encode()).hexdigest()[:12]
                claims.add(claim_hash)
                self._add_claim_node(claim_hash, content)
                
                # Add MADE_CLAIM relationship
                if author != "unknown":
                    self._add_edge(author, claim_hash, "MADE_CLAIM", evidence.get("timestamp"))
        
        # Analyze the network
        analysis = self._analyze_network()
        
        return {
            "network_stats": {
                "total_accounts": len(accounts),
                "total_media": len(media_items),
                "total_claims": len(claims),
                "total_edges": len(self.graph.edges()),
                "total_nodes": len(self.graph.nodes())
            },
            "analysis": analysis,
            "graph_data": self._export_graph_data()
        }
    
    def _add_account_node(self, account_id: str, evidence_data: Dict):
        """Add an account node with attributes"""
        node_attrs = {
            "type": "account",
            "label": account_id,
            "trust_score": evidence_data.get("trust_score", 0.5),
            "post_count": 1 if evidence_data else 0
        }
        
        if self.graph.has_node(account_id):
            # Update existing node
            existing_attrs = self.graph.nodes[account_id]
            existing_attrs["post_count"] = existing_attrs.get("post_count", 0) + 1
        else:
            self.graph.add_node(account_id, **node_attrs)
            self.directed_graph.add_node(account_id, **node_attrs)
    
    def _add_media_node(self, media_hash: str, media_data: Dict):
        """Add a media node with attributes"""
        node_attrs = {
            "type": "media",
            "label": f"Media_{media_hash[:8]}",
            "url": media_data.get("url", ""),
            "media_type": media_data.get("media_type", "unknown"),
            "repurposed": media_data.get("origin_analysis", {}).get("repurposed", False)
        }
        
        self.graph.add_node(media_hash, **node_attrs)
        self.directed_graph.add_node(media_hash, **node_attrs)
    
    def _add_claim_node(self, claim_hash: str, content: str):
        """Add a claim node with attributes"""
        node_attrs = {
            "type": "claim",
            "label": f"Claim_{claim_hash[:8]}",
            "content": content[:100] + "..." if len(content) > 100 else content,
            "length": len(content)
        }
        
        self.graph.add_node(claim_hash, **node_attrs)
        self.directed_graph.add_node(claim_hash, **node_attrs)
    
    def _add_edge(self, source: str, target: str, relationship: str, timestamp: str = None):
        """Add an edge between nodes"""
        edge_attrs = {
            "relationship": relationship,
            "timestamp": timestamp
        }
        
        self.graph.add_edge(source, target, **edge_attrs)
        self.directed_graph.add_edge(source, target, **edge_attrs)
    
    def _get_media_hash(self, url: str) -> str:
        """Generate a consistent hash for media URLs"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    async def _google_reverse_search(self, media_url: str) -> List[Dict]:
        """Simulate Google reverse image search using web search"""
        try:
            # Create a search query based on the media URL domain
            search_query = "conference meeting business image"
            results = await browser_search(search_query)
            
            formatted_results = []
            if results.get("success") and results.get("results"):
                for i, result in enumerate(results["results"][:3]):
                    formatted_results.append({
                        "source_url": result.get("url", ""),
                        "source_domain": result.get("url", "").split('/')[2] if result.get("url") else "",
                        "found_date": "2023-03-15" if i == 0 else "2025-08-31",  # Simulate older origin
                        "context": result.get("description", ""),
                        "confidence": 0.9 - (i * 0.1),
                        "search_type": "visual_similarity"
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in Google reverse search: {e}")
            return []
    
    async def _search_by_description(self, description: str) -> List[Dict]:
        """Search by image description"""
        try:
            results = await browser_search(description)
            
            formatted_results = []
            if results.get("success") and results.get("results"):
                for result in results["results"][:2]:
                    formatted_results.append({
                        "source_url": result.get("url", ""),
                        "source_domain": result.get("url", "").split('/')[2] if result.get("url") else "",
                        "found_date": "unknown",
                        "context": result.get("description", ""),
                        "confidence": 0.6,
                        "search_type": "description_match"
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in description search: {e}")
            return []
    
    async def _search_by_metadata(self, metadata: Dict) -> List[Dict]:
        """Search based on metadata patterns"""
        # For now, return empty - this would be implemented with specialized services
        return []
    
    async def _analyze_search_results(self, search_results: List[Dict], original_url: str) -> Dict:
        """Analyze search results to determine origin and repurposing"""
        if not search_results:
            return {
                "likely_origin": None,
                "repurposed": False,
                "confidence": 0.0,
                "evidence": "No matching images found"
            }
        
        # Sort by confidence and date
        sorted_results = sorted(search_results, key=lambda x: (x.get("confidence", 0), x.get("found_date", "")))
        
        # Look for older sources
        older_sources = [r for r in sorted_results if "2023" in r.get("found_date", "") or "2022" in r.get("found_date", "")]
        current_sources = [r for r in sorted_results if "2025" in r.get("found_date", "") or "2024" in r.get("found_date", "")]
        
        analysis = {
            "likely_origin": None,
            "repurposed": False,
            "confidence": 0.0,
            "evidence": "",
            "timeline": sorted_results
        }
        
        if older_sources and current_sources:
            # Found both old and new instances - likely repurposed
            analysis["likely_origin"] = older_sources[0]
            analysis["repurposed"] = True
            analysis["confidence"] = 0.8
            analysis["evidence"] = f"Image found in {len(older_sources)} older sources and {len(current_sources)} recent sources"
        
        elif older_sources:
            # Only found in older sources
            analysis["likely_origin"] = older_sources[0]
            analysis["confidence"] = 0.7
            analysis["evidence"] = f"Image only found in older sources from {older_sources[0].get('found_date')}"
        
        return analysis
    
    def _analyze_network(self) -> Dict:
        """Analyze the network for suspicious patterns"""
        analysis = {
            "centrality_analysis": self._calculate_centrality(),
            "clustering_analysis": self._analyze_clustering(),
            "suspicious_patterns": self._detect_suspicious_patterns(),
            "campaign_indicators": self._detect_campaign_indicators()
        }
        
        return analysis
    
    def _calculate_centrality(self) -> Dict:
        """Calculate centrality measures to identify key players"""
        if len(self.graph.nodes()) == 0:
            return {}
        
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            # Get top accounts by centrality
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "degree_centrality": dict(top_degree),
                "betweenness_centrality": dict(top_betweenness),
                "network_density": nx.density(self.graph)
            }
        except:
            return {"error": "Could not calculate centrality"}
    
    def _analyze_clustering(self) -> Dict:
        """Analyze clustering to detect coordinated groups"""
        try:
            # Get accounts only
            account_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get("type") == "account"]
            
            if len(account_nodes) < 2:
                return {"clusters": [], "coordination_score": 0.0}
            
            # Create subgraph with only accounts
            account_subgraph = self.graph.subgraph(account_nodes)
            
            # Find connected components (potential coordinated groups)
            clusters = list(nx.connected_components(account_subgraph))
            
            # Analyze each cluster
            cluster_analysis = []
            for i, cluster in enumerate(clusters):
                if len(cluster) > 1:  # Only interested in multi-account clusters
                    cluster_info = {
                        "cluster_id": i,
                        "accounts": list(cluster),
                        "size": len(cluster),
                        "internal_connections": account_subgraph.subgraph(cluster).number_of_edges(),
                        "coordination_indicators": self._analyze_cluster_coordination(cluster)
                    }
                    cluster_analysis.append(cluster_info)
            
            # Calculate overall coordination score
            coordination_score = self._calculate_coordination_score(cluster_analysis)
            
            return {
                "clusters": cluster_analysis,
                "coordination_score": coordination_score,
                "total_clusters": len(cluster_analysis)
            }
            
        except Exception as e:
            return {"error": f"Clustering analysis failed: {str(e)}"}
    
    def _analyze_cluster_coordination(self, cluster: Set[str]) -> List[str]:
        """Analyze a cluster for coordination indicators"""
        indicators = []
        
        cluster_list = list(cluster)
        
        # Check for similar posting times
        posting_times = []
        for account in cluster_list:
            for _, _, edge_data in self.graph.edges(account, data=True):
                if edge_data.get("relationship") == "POSTED" and edge_data.get("timestamp"):
                    posting_times.append(edge_data["timestamp"])
        
        if len(set(posting_times)) < len(posting_times) * 0.8:  # Many similar timestamps
            indicators.append("synchronized_posting")
        
        # Check for mutual mentions
        mention_count = 0
        for account1 in cluster_list:
            for account2 in cluster_list:
                if account1 != account2 and self.graph.has_edge(account1, account2):
                    edge_data = self.graph.get_edge_data(account1, account2)
                    if edge_data.get("relationship") == "MENTIONED":
                        mention_count += 1
        
        if mention_count > len(cluster_list):  # More mentions than accounts
            indicators.append("excessive_cross_mentions")
        
        # Check for similar trust scores (indicating similar account types)
        trust_scores = []
        for account in cluster_list:
            if self.graph.has_node(account):
                trust_score = self.graph.nodes[account].get("trust_score")
                if trust_score is not None:
                    trust_scores.append(trust_score)
        
        if len(trust_scores) > 1:
            avg_trust = sum(trust_scores) / len(trust_scores)
            if avg_trust < 0.4:  # Cluster of low-trust accounts
                indicators.append("low_trust_cluster")
        
        return indicators
    
    def _calculate_coordination_score(self, cluster_analysis: List[Dict]) -> float:
        """Calculate overall coordination score for the network"""
        if not cluster_analysis:
            return 0.0
        
        score = 0.0
        total_accounts = sum(cluster["size"] for cluster in cluster_analysis)
        
        for cluster in cluster_analysis:
            cluster_score = 0.0
            
            # Points for cluster size
            if cluster["size"] >= 3:
                cluster_score += 0.3
            
            # Points for coordination indicators
            indicators = cluster["coordination_indicators"]
            if "synchronized_posting" in indicators:
                cluster_score += 0.4
            if "excessive_cross_mentions" in indicators:
                cluster_score += 0.3
            if "low_trust_cluster" in indicators:
                cluster_score += 0.3
            
            # Weight by cluster size
            weighted_score = cluster_score * (cluster["size"] / total_accounts)
            score += weighted_score
        
        return min(1.0, score)  # Cap at 1.0
    
    def _detect_suspicious_patterns(self) -> List[Dict]:
        """Detect suspicious patterns in the network"""
        patterns = []
        
        # Pattern 1: Rapid amplification (same content shared quickly)
        rapid_amplification = self._detect_rapid_amplification()
        if rapid_amplification:
            patterns.extend(rapid_amplification)
        
        # Pattern 2: Sockpuppet networks (low-trust accounts all mentioning each other)
        sockpuppets = self._detect_sockpuppet_networks()
        if sockpuppets:
            patterns.extend(sockpuppets)
        
        # Pattern 3: Media repurposing (same media used in different contexts)
        repurposed_media = self._detect_repurposed_media()
        if repurposed_media:
            patterns.extend(repurposed_media)
        
        return patterns
    
    def _detect_rapid_amplification(self) -> List[Dict]:
        """Detect rapid amplification patterns"""
        patterns = []
        
        # Group posts by media content
        media_posts = defaultdict(list)
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get("type") == "media":
                # Find all accounts that posted this media
                posters = []
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor].get("type") == "account":
                        edge_data = self.graph.get_edge_data(neighbor, node)
                        if edge_data.get("relationship") == "POSTED":
                            posters.append({
                                "account": neighbor,
                                "timestamp": edge_data.get("timestamp")
                            })
                
                if len(posters) > 1:
                    media_posts[node] = posters
        
        # Analyze timing of posts
        for media_node, posters in media_posts.items():
            if len(posters) >= 2:
                # Check if posts were made within a short time window
                timestamps = [p["timestamp"] for p in posters if p["timestamp"]]
                if len(timestamps) >= 2:
                    # Simple heuristic: if multiple posts within same hour
                    unique_hours = set(ts[:13] for ts in timestamps)  # YYYY-MM-DDTHH
                    if len(unique_hours) == 1:
                        patterns.append({
                            "pattern_type": "rapid_amplification",
                            "description": f"Media {media_node[:8]} was posted by {len(posters)} accounts within the same hour",
                            "accounts_involved": [p["account"] for p in posters],
                            "media_node": media_node,
                            "severity": "HIGH" if len(posters) >= 3 else "MODERATE"
                        })
        
        return patterns
    
    def _detect_sockpuppet_networks(self) -> List[Dict]:
        """Detect potential sockpuppet networks"""
        patterns = []
        
        # Find accounts with low trust scores that mention each other frequently
        low_trust_accounts = []
        for node in self.graph.nodes():
            if (self.graph.nodes[node].get("type") == "account" and 
                self.graph.nodes[node].get("trust_score", 1.0) < 0.5):
                low_trust_accounts.append(node)
        
        if len(low_trust_accounts) >= 2:
            # Check for mutual mentions among low-trust accounts
            mention_network = []
            for account1 in low_trust_accounts:
                for account2 in low_trust_accounts:
                    if (account1 != account2 and 
                        self.graph.has_edge(account1, account2)):
                        edge_data = self.graph.get_edge_data(account1, account2)
                        if edge_data.get("relationship") == "MENTIONED":
                            mention_network.append((account1, account2))
            
            if len(mention_network) >= 2:
                patterns.append({
                    "pattern_type": "sockpuppet_network",
                    "description": f"Network of {len(low_trust_accounts)} low-trust accounts with {len(mention_network)} cross-mentions",
                    "accounts_involved": low_trust_accounts,
                    "mention_connections": mention_network,
                    "severity": "HIGH" if len(low_trust_accounts) >= 4 else "MODERATE"
                })
        
        return patterns
    
    def _detect_repurposed_media(self) -> List[Dict]:
        """Detect repurposed media artifacts"""
        patterns = []
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get("type") == "media":
                node_data = self.graph.nodes[node]
                if node_data.get("repurposed"):
                    # Find all accounts that used this repurposed media
                    users = []
                    for neighbor in self.graph.neighbors(node):
                        if self.graph.nodes[neighbor].get("type") == "account":
                            edge_data = self.graph.get_edge_data(neighbor, node)
                            if edge_data.get("relationship") == "POSTED":
                                users.append(neighbor)
                    
                    patterns.append({
                        "pattern_type": "repurposed_media",
                        "description": f"Media {node[:8]} is repurposed content used by {len(users)} accounts",
                        "media_node": node,
                        "accounts_involved": users,
                        "severity": "HIGH" if len(users) > 1 else "MODERATE"
                    })
        
        return patterns
    
    def _detect_campaign_indicators(self) -> Dict:
        """Detect indicators of coordinated campaigns"""
        indicators = {
            "coordinated_timing": False,
            "hashtag_coordination": False,
            "narrative_consistency": False,
            "amplification_network": False
        }
        
        # Check for coordinated timing
        all_timestamps = []
        for _, _, edge_data in self.graph.edges(data=True):
            if edge_data.get("timestamp"):
                all_timestamps.append(edge_data["timestamp"])
        
        if len(all_timestamps) > 2:
            # Simple heuristic: if many posts within same 2-hour window
            hour_groups = defaultdict(int)
            for ts in all_timestamps:
                hour_key = ts[:13]  # YYYY-MM-DDTHH
                hour_groups[hour_key] += 1
            
            max_posts_per_hour = max(hour_groups.values()) if hour_groups else 0
            if max_posts_per_hour >= 3:
                indicators["coordinated_timing"] = True
        
        # Check for amplification network (accounts mentioning VIP targets)
        vip_mentions = 0
        total_mentions = 0
        for _, _, edge_data in self.graph.edges(data=True):
            if edge_data.get("relationship") == "MENTIONED":
                total_mentions += 1
                # In a real implementation, you'd check against VIP target list
                if any(vip in edge_data.get("target", "") for vip in ["@VerisTruth", "@VerisProject"]):
                    vip_mentions += 1
        
        if total_mentions > 0 and vip_mentions / total_mentions > 0.5:
            indicators["amplification_network"] = True
        
        return indicators
    
    def _export_graph_data(self) -> Dict:
        """Export graph data for visualization"""
        nodes = []
        edges = []
        
        # Export nodes
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id].copy()
            node_data["id"] = node_id
            nodes.append(node_data)
        
        # Export edges
        for source, target in self.graph.edges():
            edge_data = self.graph.get_edge_data(source, target).copy()
            edge_data["source"] = source
            edge_data["target"] = target
            edges.append(edge_data)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "graph_type": "investigation_network"
        }
    
    def generate_investigation_report(self, evidence_filepath: str = "collected_evidence.json") -> Dict:
        """Generate a comprehensive investigation report"""
        # Load evidence and build network
        evidence_data = self.load_evidence_from_file(evidence_filepath)
        network_analysis = self.build_network_from_evidence(evidence_data)
        
        # Generate executive summary
        summary = self._generate_executive_summary(network_analysis, evidence_data)
        
        return {
            "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": summary,
            "network_analysis": network_analysis,
            "evidence_count": len(evidence_data.get("evidence", [])),
            "recommendations": self._generate_recommendations(network_analysis)
        }
    
    def _generate_executive_summary(self, network_analysis: Dict, evidence_data: Dict) -> Dict:
        """Generate executive summary of findings"""
        analysis = network_analysis.get("analysis", {})
        suspicious_patterns = analysis.get("suspicious_patterns", [])
        campaign_indicators = analysis.get("campaign_indicators", {})
        
        # Determine threat level
        threat_level = "LOW"
        if len(suspicious_patterns) >= 2:
            threat_level = "MODERATE"
        if len(suspicious_patterns) >= 3 or any(p.get("severity") == "HIGH" for p in suspicious_patterns):
            threat_level = "HIGH"
        
        # Count coordination indicators
        coordination_count = sum(1 for v in campaign_indicators.values() if v)
        
        summary = {
            "threat_level": threat_level,
            "coordination_detected": coordination_count > 2,
            "suspicious_patterns_found": len(suspicious_patterns),
            "key_findings": [],
            "primary_actors": []
        }
        
        # Extract key findings
        for pattern in suspicious_patterns:
            summary["key_findings"].append({
                "type": pattern["pattern_type"],
                "description": pattern["description"],
                "severity": pattern.get("severity", "UNKNOWN")
            })
        
        # Extract primary actors from centrality analysis
        centrality = analysis.get("centrality_analysis", {})
        if centrality.get("degree_centrality"):
            top_actors = list(centrality["degree_centrality"].keys())[:3]
            summary["primary_actors"] = top_actors
        
        return summary
    
    def _generate_recommendations(self, network_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        analysis = network_analysis.get("analysis", {})
        suspicious_patterns = analysis.get("suspicious_patterns", [])
        
        # Recommendations based on patterns found
        pattern_types = {p["pattern_type"] for p in suspicious_patterns}
        
        if "rapid_amplification" in pattern_types:
            recommendations.append("Monitor accounts involved in rapid amplification for coordinated behavior")
        
        if "sockpuppet_network" in pattern_types:
            recommendations.append("Investigate low-trust account cluster for potential bot network")
        
        if "repurposed_media" in pattern_types:
            recommendations.append("Verify the original source and context of repurposed media")
        
        # General recommendations
        centrality = analysis.get("centrality_analysis", {})
        if centrality.get("degree_centrality"):
            top_actor = list(centrality["degree_centrality"].keys())[0]
            recommendations.append(f"Focus monitoring efforts on high-centrality account: {top_actor}")
        
        if not recommendations:
            recommendations.append("Continue monitoring for emerging patterns")
        
        return recommendations
    
    def visualize_network(self, output_filename: str = "investigation_network.png"):
        """Create a visual representation of the network"""
        try:
            if len(self.graph.nodes()) == 0:
                print("No network data to visualize")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Create layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Separate nodes by type for different colors
            account_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get("type") == "account"]
            media_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get("type") == "media"]
            claim_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get("type") == "claim"]
            
            # Draw nodes with different colors
            if account_nodes:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=account_nodes, 
                                     node_color='lightblue', node_size=500, alpha=0.7)
            if media_nodes:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=media_nodes, 
                                     node_color='lightcoral', node_size=300, alpha=0.7)
            if claim_nodes:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=claim_nodes, 
                                     node_color='lightgreen', node_size=400, alpha=0.7)
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, alpha=0.5, width=1)
            
            # Add labels
            labels = {node: self.graph.nodes[node].get("label", node[:10]) for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            # Create legend
            legend_elements = [
                mpatches.Patch(color='lightblue', label='Accounts'),
                mpatches.Patch(color='lightcoral', label='Media'),
                mpatches.Patch(color='lightgreen', label='Claims')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title("Investigation Network Graph")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Network visualization saved to {output_filename}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

# Integration function to enhance existing workflow
def analyze_collected_evidence(evidence_filepath: str = "collected_evidence.json") -> Dict:
    """
    Main function to analyze collected evidence and generate network insights
    """
    analyzer = NetworkGraphAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_investigation_report(evidence_filepath)
    
    # Create visualization
    analyzer.visualize_network()
    
    # Save analysis results
    analysis_filename = f"network_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Network analysis saved to {analysis_filename}")
    
    return report

# Example usage and testing
async def test_network_analysis():
    """Test the network analysis with sample data"""
    
    # Create sample evidence data for testing
    sample_evidence = {
        "evidence": [
            {
                "evidence_id": "x_f3206dd5",
                "source_type": "twitter",
                "content": "Just came across @VerisTruth seems like they're working on tracking online info #VerisProject",
                "timestamp": "2025-08-31T00:02:14Z",
                "author_id": "@aryan36007",
                "trust_score": 0.3,
                "mentioned_accounts": ["@VerisTruth"],
                "hashtags": ["#VerisProject"],
                "media": [
                    {
                        "media_id": "media_gZmyP_saAAAZX_d",
                        "media_type": "image",
                        "url": "https://pbs.twimg.com/media/GzmyP_saAAAZX_d?format=jpg&name=small"
                    }
                ]
            },
            {
                "evidence_id": "x_amplifier1",
                "source_type": "twitter", 
                "content": "Sharing important info about @VerisTruth data issues #VerisProject",
                "timestamp": "2025-08-31T00:05:14Z",
                "author_id": "@amplifier_bot1",
                "trust_score": 0.2,
                "mentioned_accounts": ["@VerisTruth"],
                "hashtags": ["#VerisProject"]
            }
        ]
    }
    
    # Save sample data
    with open("test_evidence.json", 'w') as f:
        json.dump(sample_evidence, f, indent=2)
    
    # Run analysis
    analyzer = NetworkGraphAnalyzer()
    report = analyzer.generate_investigation_report("test_evidence.json")
    
    print("Network Analysis Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_network_analysis())
