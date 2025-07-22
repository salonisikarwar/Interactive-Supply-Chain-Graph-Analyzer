import networkx as nx
import pandas as pd
import random
import heapq
import matplotlib.pyplot as plt
import time
import os

class SupplyChainGraph:
    def __init__(self, csv_file=None, num_nodes=None, edge_prob=0.01):
        """
        Initialize a new directed graph.
        - csv_file: Path to CSV file for loading (optional).
        - num_nodes: Number of nodes for random graph generation (optional).
        - edge_prob: Probability of an edge between any two nodes (default 0.01).
        """
        self.graph = nx.DiGraph()
        self.removed_nodes = {}  # Store removed nodes and their properties
        self.removed_edges = {}  # Store removed edges and their properties (from_node, to_node: cost)
        if csv_file:
            self.load_graph(csv_file)
        elif num_nodes is not None:
            self.generate_random_network(num_nodes, edge_prob)

    def load_graph(self, csv_file):
        """Load directed graph from a CSV file with columns: source, destination, cost."""
        try:
            data = pd.read_csv(csv_file)
            if data.empty:
                print("Warning: CSV file is empty. Initializing empty graph.")
                return
            data['cost'] = pd.to_numeric(data['cost'], errors='coerce')
            print("Graph data preview:\n", data.head())
            self.graph = nx.from_pandas_edgelist(
                data,
                source='source',
                target='destination',
                edge_attr='cost',
                create_using=nx.DiGraph
            )
            for u, v, data in self.graph.edges(data=True):
                if 'cost' not in data or pd.isna(data['cost']):
                    self.graph[u][v]['cost'] = 1.0  # Default cost
            print(f"Loaded {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}. Initializing empty graph.")
        except Exception as e:
            print(f"Error loading CSV: {e}. Initializing empty graph.")

    def generate_random_network(self, num_nodes, edge_prob=0.01, cost_range=(1.0, 100.0)):
        """Generate a random directed supply chain network."""
        node_types = ['Supplier', 'Warehouse', 'Retailer']
        nodes = [f"{t}_{i+1}" for t in node_types for i in range(num_nodes // len(node_types))]
        nodes.extend([f"{node_types[i % len(node_types)]}_{num_nodes // len(node_types) + 1 + i}" 
                     for i in range(num_nodes % len(node_types))])
        random.shuffle(nodes)
        self.graph.add_nodes_from(nodes)
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < edge_prob:
                    cost = random.uniform(cost_range[0], cost_range[1])
                    edges.append((nodes[i], nodes[j], {'cost': cost}))
        self.graph.add_edges_from(edges)
        print(f"Generated {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges "
              f"with edge probability {edge_prob} and cost range {cost_range}.")

    def add_node(self, node):
        """Add a node if it doesn't exist."""
        if not self.graph.has_node(node):
            self.graph.add_node(node)
            if node in self.removed_nodes:
                del self.removed_nodes[node]

    def add_edge(self, from_node, to_node, cost=1.0):
        """Add a directed edge with cost."""
        self.add_node(from_node)
        self.add_node(to_node)
        if not self.graph.has_edge(from_node, to_node):
            self.graph.add_edge(from_node, to_node, cost=cost)
            if (from_node, to_node) in self.removed_edges:
                del self.removed_edges[(from_node, to_node)]

    def update_edge_cost(self, from_node, to_node, new_cost):
        """Update the cost of an existing edge."""
        if self.graph.has_edge(from_node, to_node):
            if not isinstance(new_cost, (int, float)) or new_cost < 0:
                print(f"Error: New cost must be a non-negative number, got {new_cost}.")
                return
            self.graph[from_node][to_node]['cost'] = new_cost
            print(f"Updated cost of edge {from_node} -> {to_node} to {new_cost}.")
        else:
            print(f"Error: Edge {from_node} -> {to_node} does not exist.")

    def remove_node(self, node):
        """Remove a node and its associated edges, storing them for potential restoration."""
        if self.graph.has_node(node):
            for neighbor in list(self.graph.neighbors(node)):
                if self.graph.has_edge(node, neighbor):
                    self.removed_edges[(node, neighbor)] = self.graph[node][neighbor]['cost']
            for predecessor in list(self.graph.predecessors(node)):
                if self.graph.has_edge(predecessor, node):
                    self.removed_edges[(predecessor, node)] = self.graph[predecessor][node]['cost']
            self.removed_nodes[node] = {}
            self.graph.remove_node(node)
            print(f"Removed node {node} and its edges.")

    def restore_node(self, node):
        """Restore a previously removed node and its edges."""
        if node in self.removed_nodes:
            self.add_node(node)
            for (src, tgt), cost in list(self.removed_edges.items()):
                if src == node or tgt == node:
                    if not self.graph.has_edge(src, tgt):
                        self.add_edge(src, tgt, cost)
            edges_to_remove = [(src, tgt) for (src, tgt), _ in self.removed_edges.items() 
                              if src == node or tgt == node]
            for edge in edges_to_remove:
                if (edge in self.removed_edges and not self.graph.has_edge(edge[0], edge[1])):
                    del self.removed_edges[edge]
            print(f"Restored node {node} and its edges.")
        else:
            print(f"Node {node} was not previously removed.")

    def remove_edge(self, from_node, to_node):
        """Remove a specific edge, storing its cost for potential restoration."""
        if self.graph.has_edge(from_node, to_node):
            cost = self.graph[from_node][to_node]['cost']
            self.removed_edges[(from_node, to_node)] = cost
            self.graph.remove_edge(from_node, to_node)
            print(f"Removed edge {from_node} -> {to_node} with cost {cost}.")

    def restore_edge(self, from_node, to_node):
        """Restore a previously removed edge with its original cost."""
        if (from_node, to_node) in self.removed_edges:
            cost = self.removed_edges[(from_node, to_node)]
            self.add_edge(from_node, to_node, cost)
            if (from_node, to_node) in self.removed_edges and not self.graph.has_edge(from_node, to_node):
                del self.removed_edges[(from_node, to_node)]
            print(f"Restored edge {from_node} -> {to_node} with cost {cost}.")
        else:
            print(f"Edge {from_node} -> {to_node} was not previously removed.")

    def find_least_cost_path(self, start_node, end_node):
        """Find the least-cost path between start_node and end_node using Dijkstra's algorithm."""
        if not self.graph.has_node(start_node) or not self.graph.has_node(end_node):
            print(f"Error: Start node {start_node} or end node {end_node} does not exist.")
            return None

        if not nx.is_strongly_connected(self.graph):
            print("Warning: Graph is not fully connected. Path may not exist.")
        
        distances = {node: float('infinity') for node in self.graph.nodes()}
        distances[start_node] = 0
        previous = {node: None for node in self.graph.nodes()}
        pq = [(0, start_node)]  # (distance, node)
        visited = set()

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == end_node:
                break

            for neighbor, data in self.graph[current_node].items():
                if neighbor not in visited:
                    distance = current_distance + data['cost']

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (distance, neighbor))

        if distances[end_node] == float('infinity'):
            print(f"No path exists between {start_node} and {end_node} due to disconnection.")
            return None

        # Reconstruct path
        path = []
        current_node = end_node
        while current_node is not None:
            path.append(current_node)
            current_node = previous[current_node]
        path.reverse()

        print(f"Least-cost path from {start_node} to {end_node}: {' -> '.join(path)}")
        print(f"Total cost: {distances[end_node]}")
        return path, distances[end_node]

    def get_neighbors(self, node):
        """Return outgoing neighbors and their costs for a node."""
        return {neighbor: self.graph[node][neighbor]['cost']
                for neighbor in self.graph.neighbors(node) if self.graph.has_node(node)}

    def get_nodes(self):
        """Return all nodes in the graph."""
        return list(self.graph.nodes())

    def get_edge_weight(self, from_node, to_node):
        """Return the cost of a specific edge."""
        if self.graph.has_edge(from_node, to_node):
            return self.graph[from_node][to_node]['cost']
        return None

    def detect_bottlenecks(self, max_samples=10):
        """
        Detect edges that, if removed, would disconnect the graph or significantly increase path costs.
        Uses caching to avoid redundant path calculations.
        Returns a list of critical edges with their impact scores.
        """
        if not self.graph.edges():
            print("Warning: Graph has no edges. No bottlenecks to detect.")
            return []

        bottlenecks = []
        original_edges = list(self.graph.edges(data=True))
        path_cache = {}  # Cache for shortest path costs

        def get_path_cost(start, end):
            if (start, end) not in path_cache:
                path_result = self.find_least_cost_path(start, end)
                path_cache[(start, end)] = path_result[1] if path_result else float('infinity')
            return path_cache[(start, end)]

        for u, v, data in original_edges:
            # Temporarily remove edge
            self.graph.remove_edge(u, v)
            
            # Check if graph becomes disconnected
            strongly_connected = list(nx.strongly_connected_components(self.graph))
            is_disconnected = len(strongly_connected) > 1 or not nx.is_strongly_connected(self.graph)
            
            # Calculate impact on shortest paths
            cost_impact = 0
            sample_pairs = [(s, t) for s in self.graph.nodes() for t in self.graph.nodes() 
                           if s != t and nx.has_path(self.graph, s, t)][:max_samples]
            if not sample_pairs:
                cost_impact = float('infinity')
            else:
                for start, end in sample_pairs:
                    cost_impact += get_path_cost(start, end)
            
            # Restore edge
            self.graph.add_edge(u, v, cost=data['cost'])
            
            # Store bottleneck info
            impact_score = float('infinity') if is_disconnected else cost_impact
            bottlenecks.append(((u, v), impact_score, is_disconnected))
        
        # Sort by impact score
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        if bottlenecks:
            print("Potential bottlenecks (edge, impact_score, causes_disconnect):")
            for edge, score, disconnect in bottlenecks[:5]:  # Top 5 bottlenecks
                print(f"Edge {edge}: Impact Score = {score:.2f}, Disconnects = {disconnect}")
        else:
            print("No bottlenecks detected due to empty or fully disconnected graph.")
        return bottlenecks

    def find_isolated_nodes(self):
        """
        Identify nodes with no incoming or outgoing edges.
        Returns a list of isolated nodes.
        """
        isolated = [node for node in self.graph.nodes() 
                   if self.graph.in_degree(node) == 0 and self.graph.out_degree(node) == 0]
        if isolated:
            print(f"Isolated nodes found: {isolated}")
        else:
            print("No isolated nodes found.")
        return isolated

    def find_disconnected_components(self):
        """
        Identify strongly connected components in the graph.
        Returns a list of sets, each containing nodes in a component.
        """
        components = list(nx.strongly_connected_components(self.graph))
        if components:
            print(f"Found {len(components)} strongly connected components:")
            for i, component in enumerate(components, 1):
                print(f"Component {i}: {component}")
        else:
            print("No components found due to empty graph.")
        return components

    def network_health_status(self):
        """
        Provide a summary of network health metrics.
        Returns a dictionary with key metrics.
        """
        if not self.graph.nodes():
            print("Warning: Graph is empty. Returning default health metrics.")
            return {
                'num_nodes': 0, 'num_edges': 0, 'num_isolated_nodes': 0,
                'num_components': 0, 'num_potential_bottlenecks': 0,
                'avg_clustering_coefficient': 0.0, 'avg_path_length': float('infinity')
            }

        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        isolated_nodes = self.find_isolated_nodes()
        components = self.find_disconnected_components()
        bottlenecks = self.detect_bottlenecks()
        
        # Calculate average clustering coefficient
        avg_clustering = nx.average_clustering(self.graph.to_undirected()) if num_edges > 0 else 0.0
        
        # Calculate average shortest path length (for connected nodes)
        avg_path_length = 0
        connected_pairs = 0
        for s in self.graph.nodes():
            for t in self.graph.nodes():
                if s != t and nx.has_path(self.graph, s, t):
                    path, cost = self.find_least_cost_path(s, t) if self.find_least_cost_path(s, t) else ([], float('infinity'))
                    if cost != float('infinity'):
                        avg_path_length += cost
                        connected_pairs += 1
        avg_path_length = avg_path_length / connected_pairs if connected_pairs > 0 else float('infinity')
        
        health_metrics = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_isolated_nodes': len(isolated_nodes),
            'num_components': len(components) if components else 0,
            'num_potential_bottlenecks': len([b for b in bottlenecks if b[2] or b[1] > 100]),
            'avg_clustering_coefficient': avg_clustering,
            'avg_path_length': avg_path_length
        }
        
        print("\nNetwork Health Status:")
        print(f"Nodes: {health_metrics['num_nodes']}, Edges: {health_metrics['num_edges']}")
        print(f"Isolated Nodes: {health_metrics['num_isolated_nodes']}")
        print(f"Strongly Connected Components: {health_metrics['num_components']}")
        print(f"Potential Bottlenecks: {health_metrics['num_potential_bottlenecks']}")
        print(f"Average Clustering Coefficient: {health_metrics['avg_clustering_coefficient']:.3f}")
        print(f"Average Path Length: {health_metrics['avg_path_length']:.2f}")
        
        return health_metrics

    def visualize_graph(self, highlight_bottlenecks=True, highlight_path=None):
        """
        Visualize the graph using Matplotlib, highlighting bottlenecks and isolated nodes.
        - highlight_bottlenecks: If True, critical edges are highlighted in red.
        - highlight_path: Tuple (start_node, end_node) to highlight the least-cost path.
        """
        if not self.graph.nodes():
            print("Warning: Graph is empty. No visualization possible.")
            return

        plt.figure(figsize=(12, 10))  # Larger figure for bigger graphs
        pos = nx.circular_layout(self.graph)
        
        # Color nodes by type
        node_colors = []
        for node in self.graph.nodes():
            if 'Supplier' in node:
                node_colors.append('#1f77b4')  # Blue for suppliers
            elif 'Warehouse' in node:
                node_colors.append('#2ca02c')  # Green for warehouses
            elif 'Retailer' in node:
                node_colors.append('#ff7f0e')  # Orange for retailers
            else:
                node_colors.append('#7f7f7f')  # Gray for others
        
        # Highlight isolated nodes
        isolated = self.find_isolated_nodes()
        for i, node in enumerate(self.graph.nodes()):
            if node in isolated:
                node_colors[i] = '#d62728'  # Red for isolated nodes
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=300)  # Reduced size for scalability
        nx.draw_networkx_labels(self.graph, pos, font_size=6)  # Smaller font for readability
        
        # Draw edges
        edge_colors = []
        edge_labels = {}
        bottlenecks = [b[0] for b in self.detect_bottlenecks() if b[2]] if highlight_bottlenecks else []
        path_edges = set()
        if highlight_path:
            start, end = highlight_path
            path, _ = self.find_least_cost_path(start, end) if self.find_least_cost_path(start, end) else ([], None)
            if path and len(path) > 1:
                path_edges = set(zip(path[:-1], path[1:]))
        
        for u, v in self.graph.edges():
            is_bottleneck = (u, v) in bottlenecks
            is_path = (u, v) in path_edges or (v, u) in path_edges
            edge_colors.append('#d62728' if is_bottleneck else '#ff9896' if is_path else '#7f7f7f')
            edge_labels[(u, v)] = f"{self.graph[u][v]['cost']:.1f}"
        
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, arrows=True)
        # Skip edge labels for large graphs to avoid clutter
        # nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Supplier', markerfacecolor='#1f77b4', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Warehouse', markerfacecolor='#2ca02c', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Retailer', markerfacecolor='#ff7f0e', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Isolated', markerfacecolor='#d62728', markersize=10),
            Line2D([0], [0], color='#d62728', lw=2, label='Bottleneck'),
            Line2D([0], [0], color='#ff9896', lw=2, label='Highlighted Path')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.title("Supply Chain Network Visualization")
        plt.axis('off')
        plt.show()

    def generate_bottleneck_chart(self):
        """
        Generate a Chart.js bar chart to display bottleneck impact scores.
        """
        bottlenecks = self.detect_bottlenecks()
        if not bottlenecks:
            print("No bottlenecks to chart due to empty or disconnected graph.")
            return

        labels = [f"{edge[0]}->{edge[1]}" for edge, _, _ in bottlenecks[:5]]
        impact_scores = [score if score != float('infinity') else 9999 for _, score, _ in bottlenecks[:5]]  # Cap infinity

        chart_code = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <canvas id="bottleneckChart" width="400" height="200"></canvas>
    <script>
        const ctx = document.getElementById('bottleneckChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: """ + str(labels) + """,
                datasets: [{
                    label: 'Impact Score',
                    data: """ + str(impact_scores) + """,
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: { y: { beginAtZero: true } },
                plugins: { legend: { position: 'top' } }
            }
        });
    </script>
</body>
</html>
"""
        # Save to Visualization/ directory
        chart_path = os.path.join("Visualization", "bottleneck_chart.html")
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        with open(chart_path, 'w') as f:
            f.write(chart_code)
        print(f"Bottleneck chart saved as {chart_path}. Open in a browser to view.")

    def simulate_failure(self, element, element_type='node'):
        """
        Simulate failure of a node or edge and assess network resilience.
        - element: Node or edge (tuple of from_node, to_node) to remove.
        - element_type: 'node' or 'edge'.
        """
        original_state = self.graph.copy()
        if element_type == 'node':
            if self.graph.has_node(element):
                self.remove_node(element)
                print(f"Simulated failure of node {element}.")
            else:
                print(f"Error: Node {element} does not exist.")
                return
        elif element_type == 'edge':
            if len(element) == 2 and self.graph.has_edge(element[0], element[1]):
                self.remove_edge(element[0], element[1])
                print(f"Simulated failure of edge {element[0]} -> {element[1]}.")
            else:
                print(f"Error: Edge {element} does not exist.")
                return
        else:
            print(f"Error: Invalid element_type. Use 'node' or 'edge'.")
            return

        # Assess resilience
        health = self.network_health_status()
        isolated = self.find_isolated_nodes()
        components = self.find_disconnected_components()
        print(f"Post-failure Health: Nodes {health['num_nodes']}, Edges {health['num_edges']}")
        print(f"Isolated Nodes: {isolated}")
        print(f"Components: {len(components)}")

        # Restore original state
        self.graph = original_state
        print(f"Restored original graph state: {self}")

    def suggest_reconnections(self, max_suggestions=3):
        """
        Suggest edges to reconnect isolated nodes with the lowest cost options.
        Returns a list of suggested (from_node, to_node, cost) tuples.
        """
        isolated_nodes = self.find_isolated_nodes()
        if not isolated_nodes:
            print("No isolated nodes to reconnect.")
            return []

        suggestions = []
        for isolated in isolated_nodes:
            # Find potential source nodes (non-isolated with outgoing edges)
            sources = [node for node in self.graph.nodes() if node not in isolated_nodes and self.graph.out_degree(node) > 0]
            if not sources:
                print(f"No sources available to reconnect {isolated}.")
                continue
            for source in sources:
                # Suggest a cost based on the average of existing edges from the source
                outgoing_costs = [self.graph[source][neighbor]['cost'] for neighbor in self.graph.neighbors(source)]
                suggested_cost = sum(outgoing_costs) / len(outgoing_costs) if outgoing_costs else 10.0  # Default cost if none
                suggestions.append((source, isolated, suggested_cost))

        # Sort by suggested cost
        suggestions.sort(key=lambda x: x[2])
        if suggestions:
            print(f"Suggested reconnections for isolated nodes {isolated_nodes}:")
            for source, target, cost in suggestions[:max_suggestions]:
                print(f"Add edge {source} -> {target} with cost {cost:.1f}")
        else:
            print("No reconnection suggestions due to insufficient network structure.")
        return suggestions[:max_suggestions]

    def __str__(self):
        return f"Nodes: {len(self.graph.nodes)}, Edges: {len(self.graph.edges)}"

if __name__ == "__main__":
    import pandas as pd
    # Create data and Visualization directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("Visualization", exist_ok=True)

    # Example 1: Load from CSV (small graph)
    sample_data = pd.DataFrame({
        'source': ['Supplier_1', 'Supplier_1', 'Warehouse_A', 'Warehouse_B'],
        'destination': ['Warehouse_A', 'Warehouse_B', 'Retailer_X', 'Retailer_Y'],
        'cost': [10.5, 15.0, 5.0, 7.5]
    })
    # Save to data/ directory
    sample_path = os.path.join("data", "sample_graph.csv")
    sample_data.to_csv(sample_path, index=False)
    csv_graph = SupplyChainGraph(sample_path)
    print("Initial CSV-loaded graph:")
    print(f"Neighbors of Supplier_1: {csv_graph.get_neighbors('Supplier_1')}")
    print(csv_graph)

    # Example 2: Generate and save supply_chain.csv
    supply_chain_data = pd.DataFrame({
        'source': ['Supplier_2', 'Supplier_2', 'Warehouse_C', 'Warehouse_D'],
        'destination': ['Warehouse_C', 'Warehouse_D', 'Retailer_Z', 'Retailer_W'],
        'cost': [12.0, 18.5, 6.0, 8.0]
    })
    supply_chain_path = os.path.join("data", "supply_chain.csv")
    supply_chain_data.to_csv(supply_chain_path, index=False)
    print(f"Generated and saved supply_chain.csv to {supply_chain_path}")

    # Test with 100-node random graph
    print("\nTesting with 100-node random graph...")
    start_time = time.time()
    large_graph = SupplyChainGraph(num_nodes=100, edge_prob=0.05)
    print(f"Graph generation time: {time.time() - start_time:.2f} seconds")
    print(large_graph)

    # Test path finding
    start_time = time.time()
    nodes = large_graph.get_nodes()
    if nodes:
        start_node = nodes[0]
        end_node = nodes[-1]
        print(f"\nFinding least-cost path from {start_node} to {end_node}...")
        large_graph.find_least_cost_path(start_node, end_node)
        print(f"Path finding time: {time.time() - start_time:.2f} seconds")

    # Test bottleneck detection
    start_time = time.time()
    print("\nDetecting bottlenecks...")
    large_graph.detect_bottlenecks(max_samples=20)  # Increased samples for larger graph
    print(f"Bottleneck detection time: {time.time() - start_time:.2f} seconds")

    # Test visualization (with a sample path)
    start_time = time.time()
    print("\nVisualizing the graph...")
    if nodes:
        large_graph.visualize_graph(highlight_path=(start_node, end_node))
    print(f"Visualization time: {time.time() - start_time:.2f} seconds")

    # Test failure simulation
    start_time = time.time()
    print("\nSimulating failure of a random node...")
    if nodes:
        random_node = random.choice(nodes)
        large_graph.simulate_failure(random_node, 'node')
    print(f"Failure simulation time: {time.time() - start_time:.2f} seconds")

    # Test reconnection suggestions
    start_time = time.time()
    print("\nSuggesting reconnections...")
    large_graph.suggest_reconnections()
    print(f"Reconnection suggestion time: {time.time() - start_time:.2f} seconds")

    # Generate bottleneck chart
    start_time = time.time()
    print("\nGenerating bottleneck chart...")
    large_graph.generate_bottleneck_chart()
    print(f"Chart generation time: {time.time() - start_time:.2f} seconds")