# network_graph_analyzer.py - Refactored
import json
import networkx as nx
from datetime import datetime
from typing import Dict, List
from langchain_core.tools import tool
import hashlib
from schema import EvidenceItem, SourceType, MediaItem

class NetworkGraphAnalyzer:
    def __init__(self):
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()

    def load_evidence_from_file(self, filepath: str = "collected_evidence.json") -> Dict:
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

    def _add_account_node(self, author: str, evidence: Dict):
        self.graph.add_node(author, type="account", label=author)

    def _add_media_node(self, media_hash: str, media: Dict):
        self.graph.add_node(media_hash, type="media", label=f"Media_{media_hash[:6]}")

    def _add_claim_node(self, claim_hash: str, content: str):
        self.graph.add_node(claim_hash, type="claim", label=f"Claim_{claim_hash[:6]}")

    def _add_edge(self, source: str, target: str, relationship: str, timestamp: str):
        self.graph.add_edge(source, target, relationship=relationship, timestamp=timestamp)
        self.directed_graph.add_edge(source, target, relationship=relationship, timestamp=timestamp)

    def _analyze_network(self) -> Dict:
        analysis = {
            "centrality": nx.degree_centrality(self.graph),
            "density": nx.density(self.graph),
            "communities": list(nx.connected_components(self.graph)) if not nx.is_directed(self.graph) else []
        }
        return analysis

    def visualize_network(self, output_filename: str = "network_visualization.png"):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            pos = nx.spring_layout(self.graph)
            plt.figure(figsize=(10, 8))

            nx.draw_networkx_nodes(self.graph, pos, nodelist=[n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "account"], node_color='lightblue', node_size=500, alpha=0.8)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "media"], node_color='lightcoral', node_size=300, alpha=0.8)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "claim"], node_color='lightgreen', node_size=400, alpha=0.8)
            nx.draw_networkx_edges(self.graph, pos, alpha=0.5, width=1)

            labels = {node: self.graph.nodes[node].get("label", node[:10]) for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)

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

    @tool
    def generate_investigation_report(self, evidence_filepath: str = "collected_evidence.json") -> List[EvidenceItem]:
        """
        Generate a network analysis report from collected evidence.
        Returns a list of EvidenceItem objects with analysis findings.
        """
        evidence_data = self.load_evidence_from_file(evidence_filepath)
        self.graph.clear()
        self.directed_graph.clear()

        evidence_list = evidence_data.get("evidence", [])
        evidence_items = []

        for evidence in evidence_list:
            author = evidence.get("author_id", "unknown")
            if author != "unknown":
                self._add_account_node(author, evidence)

            if "media" in evidence:
                for media in evidence["media"]:
                    media_hash = self._get_media_hash(media["url"])
                    self._add_media_node(media_hash, media)
                    if author != "unknown":
                        self._add_edge(author, media_hash, "POSTED", evidence.get("timestamp"))

            mentioned = evidence.get("mentioned_accounts", [])
            for mentioned_account in mentioned:
                self._add_account_node(mentioned_account, {})
                if author != "unknown":
                    self._add_edge(author, mentioned_account, "MENTIONED", evidence.get("timestamp"))

            content = evidence.get("content", "")
            if content:
                claim_hash = hashlib.md5(content.encode()).hexdigest()[:12]
                self._add_claim_node(claim_hash, content)
                if author != "unknown":
                    self._add_edge(author, claim_hash, "MADE_CLAIM", evidence.get("timestamp"))

            # Create evidence item for this evidence entry
            evidence_items.append(EvidenceItem(
                source_type=SourceType.NETWORK_ANALYSIS,
                url=evidence.get("url", ""),
                content=f"Network analysis for author {author}: {content[:100]}",
                timestamp=datetime.now(),
                author_id=author,
                mentioned_accounts=mentioned,
                hashtags=evidence.get("hashtags", []),
                raw_data=self._analyze_network()
            ))

        # Visualize the network
        self.visualize_network()

        return evidence_items

    def _get_media_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]

# Integration function
@tool
def analyze_collected_evidence(evidence_filepath: str = "collected_evidence.json") -> List[EvidenceItem]:
    """
    Analyze collected evidence and generate network insights.
    Returns a list of EvidenceItem objects.
    """
    analyzer = NetworkGraphAnalyzer()
    return analyzer.generate_investigation_report(evidence_filepath)

async def test_network_analysis():
    sample_evidence = {
        "evidence": [
            {
                "evidence_id": "x_f3206dd5",
                "source_type": "twitter",
                "content": "Just came across @VerisTruth seems like they're working on tracking online info #VerisProject",
                "timestamp": "2025-08-31T00:02:14Z",
                "author_id": "@aryan36007",
                "mentioned_accounts": ["@VerisTruth"],
                "hashtags": ["#VerisProject"],
                "media": [{"media_type": "image", "url": "https://pbs.twimg.com/media/GzmyP_saAAAZX_d"}]
            },
            {
                "evidence_id": "x_amplifier1",
                "source_type": "twitter",
                "content": "Sharing important info about @VerisTruth data issues #VerisProject",
                "timestamp": "2025-08-31T00:05:14Z",
                "author_id": "@amplifier_bot1",
                "mentioned_accounts": ["@VerisTruth"],
                "hashtags": ["#VerisProject"]
            }
        ]
    }
    with open("test_evidence.json", 'w') as f:
        json.dump(sample_evidence, f, indent=2)
    analyzer = NetworkGraphAnalyzer()
    report = analyzer.generate_investigation_report("test_evidence.json")
    for item in report:
        print(item.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(test_network_analysis())