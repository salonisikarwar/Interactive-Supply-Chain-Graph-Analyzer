import streamlit as st
import graph_utils
import pandas as pd
import os
import time
from pyvis.network import Network
import networkx as nx

# Set page configuration
st.set_page_config(layout="wide", page_title="Supply Chain Network Analyzer")

# Sidebar for navigation and inputs
st.sidebar.header("Supply Chain Network Analyzer")
option = st.sidebar.selectbox(
    "Choose an Action",
    ("Generate Random Graph", "Load Graph from CSV", "Find Least-Cost Path", "Detect Bottlenecks", 
     "Visualize Graph", "Simulate Disruption", "Suggest Reconnections", "Remove Node", "Remove Edge", 
     "Restore Node", "Restore Edge")
)

# Initialize session state for graph object and visualization file
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'viz_file' not in st.session_state:
    st.session_state.viz_file = None

# Ensure visualizations directory exists
os.makedirs("visualizations", exist_ok=True)

# Action handlers
if option == "Generate Random Graph":
    st.header("Generate Random Supply Chain Graph")
    num_nodes = st.slider("Number of Nodes", min_value=10, max_value=1000, value=100, step=10)
    edge_prob = st.slider("Edge Probability", 0.01, 0.1, 0.05, 0.01)
    if st.button("Generate Graph"):
        start_time = time.time()
        st.session_state.graph = graph_utils.SupplyChainGraph(num_nodes=num_nodes, edge_prob=edge_prob)
        st.success(f"Graph generated with {num_nodes} nodes and {len(st.session_state.graph.graph.edges)} edges in {time.time() - start_time:.2f} seconds!")
        st.write(st.session_state.graph)

if option == "Load Graph from CSV":
    st.header("Load Graph from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        csv_path = os.path.join("data", "uploaded_graph.csv")
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button("Load Graph"):
            start_time = time.time()
            st.session_state.graph = graph_utils.SupplyChainGraph(csv_path)
            st.success(f"Graph loaded from {csv_path} in {time.time() - start_time:.2f} seconds!")
            st.write(st.session_state.graph)

if option == "Find Least-Cost Path":
    st.header("Find Least-Cost Path")
    if st.session_state.graph is None:
        st.warning("Please generate or load a graph first.")
    else:
        nodes = st.session_state.graph.get_nodes()
        start_node = st.selectbox("Start Node", nodes, index=0 if nodes else 0)
        end_node = st.selectbox("End Node", nodes, index=len(nodes)-1 if nodes else 0)
        if st.button("Calculate Path"):
            start_time = time.time()
            result = st.session_state.graph.find_least_cost_path(start_node, end_node)
            st.success(f"Path calculated in {time.time() - start_time:.2f} seconds!")
            if result:
                path, cost = result
                st.write(f"**Path:** {' -> '.join(path)}")
                st.write(f"**Total Cost:** {cost}")

if option == "Detect Bottlenecks":
    st.header("Detect Bottlenecks")
    if st.session_state.graph is None:
        st.warning("Please generate or load a graph first.")
    else:
        max_samples = st.slider("Max Sample Pairs", min_value=5, max_value=50, value=10, step=5)
        if st.button("Detect Bottlenecks"):
            start_time = time.time()
            bottlenecks = st.session_state.graph.detect_bottlenecks(max_samples=max_samples)
            st.success(f"Bottlenecks detected in {time.time() - start_time:.2f} seconds!")
            if bottlenecks:
                df = pd.DataFrame(bottlenecks, columns=["Edge", "Impact Score", "Causes Disconnect"])
                st.dataframe(df.style.format({"Impact Score": "{:.2f}"}))  # Formatted output

if option == "Visualize Graph":
    st.header("Visualize Supply Chain Network")
    if st.session_state.graph is None:
        st.warning("Please generate or load a graph first.")
    else:
        nodes = st.session_state.graph.get_nodes()
        highlight_path = st.checkbox("Highlight Least-Cost Path")
        start_node = st.selectbox("Start Node (for Path)", nodes, index=0 if nodes else 0) if highlight_path else None
        end_node = st.selectbox("End Node (for Path)", nodes, index=len(nodes)-1 if nodes else 0) if highlight_path else None
        if st.button("Generate Visualization"):
            start_time = time.time()
            # Create PyVis network
            net = Network(height="600px", width="100%", directed=True)
            net.from_nx(st.session_state.graph.graph)
            
            # Color nodes by type
            for node in net.nodes:
                if 'Supplier' in node['id']:
                    node['color'] = '#1f77b4'
                elif 'Warehouse' in node['id']:
                    node['color'] = '#2ca02c'
                elif 'Retailer' in node['id']:
                    node['color'] = '#ff7f0e'
                else:
                    node['color'] = '#7f7f7f'
            
            # Highlight path if selected
            if highlight_path and start_node and end_node:
                path_result = st.session_state.graph.find_least_cost_path(start_node, end_node)
                if path_result:
                    path, _ = path_result
                    path_edges = list(zip(path[:-1], path[1:]))
                    for edge in net.edges:
                        if (edge['from'], edge['to']) in path_edges:
                            edge['color'] = '#ff9896'
                            edge['width'] = 3
            
            # Detect bottlenecks for highlighting
            bottlenecks = st.session_state.graph.detect_bottlenecks(max_samples=5)  # Quick check
            bottleneck_edges = [b[0] for b in bottlenecks if b[2]]
            for edge in net.edges:
                if (edge['from'], edge['to']) in bottleneck_edges:
                    edge['color'] = '#d62728'
                    edge['width'] = 3
            
            # Save and display
            viz_file = os.path.join("visualizations", f"network_{int(time.time())}.html")
            net.save_graph(viz_file)
            st.session_state.viz_file = viz_file
            st.success(f"Visualization generated in {time.time() - start_time:.2f} seconds!")
            with open(viz_file, "r") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)

if option == "Simulate Disruption":
    st.header("Simulate Disruption")
    if st.session_state.graph is None:
        st.warning("Please generate or load a graph first.")
    else:
        nodes = st.session_state.graph.get_nodes()
        element = st.selectbox("Node to Disrupt", nodes, index=0 if nodes else 0)
        if st.button("Simulate Disruption"):
            start_time = time.time()
            st.session_state.graph.simulate_failure(element, 'node')
            st.success(f"Disruption simulated in {time.time() - start_time:.2f} seconds!")
            health = st.session_state.graph.network_health_status()
            st.write("**Post-Disruption Health:**")
            st.json(health)

if option == "Suggest Reconnections":
    st.header("Suggest Reconnections")
    if st.session_state.graph is None:
        st.warning("Please generate or load a graph first.")
    else:
        max_suggestions = st.slider("Max Suggestions", min_value=1, max_value=10, value=3, step=1)
        if st.button("Suggest Reconnections"):
            start_time = time.time()
            suggestions = st.session_state.graph.suggest_reconnections(max_suggestions=max_suggestions)
            st.success(f"Suggestions generated in {time.time() - start_time:.2f} seconds!")
            if suggestions:
                df = pd.DataFrame(suggestions, columns=["Source", "Target", "Cost"])
                st.dataframe(df.style.format({"Cost": "{:.1f}"}))  # Formatted output

if option == "Remove Node":
    st.header("Remove Node")
    if st.session_state.graph is None:
        st.warning("Please generate or load a graph first.")
    else:
        nodes = st.session_state.graph.get_nodes()
        node_to_remove = st.selectbox("Select Node to Remove", nodes, index=0 if nodes else 0)
        if st.button("Remove Node"):
            start_time = time.time()
            st.session_state.graph.remove_node(node_to_remove)
            st.success(f"Node {node_to_remove} removed in {time.time() - start_time:.2f} seconds!")
            st.write(f"Updated Graph: {st.session_state.graph}")
            health = st.session_state.graph.network_health_status()
            st.write("**Updated Health Status:**")
            st.json(health)

if option == "Remove Edge":
    st.header("Remove Edge")
    if st.session_state.graph is None:
        st.warning("Please generate or load a graph first.")
    else:
        nodes = st.session_state.graph.get_nodes()
        from_node = st.selectbox("From Node", nodes, index=0 if nodes else 0)
        to_node = st.selectbox("To Node", [n for n in nodes if st.session_state.graph.get_edge_weight(from_node, n) is not None], index=0 if nodes else 0)
        if st.button("Remove Edge"):
            start_time = time.time()
            st.session_state.graph.remove_edge(from_node, to_node)
            st.success(f"Edge {from_node} -> {to_node} removed in {time.time() - start_time:.2f} seconds!")
            st.write(f"Updated Graph: {st.session_state.graph}")
            health = st.session_state.graph.network_health_status()
            st.write("**Updated Health Status:**")
            st.json(health)

if option == "Restore Node":
    st.header("Restore Node")
    if st.session_state.graph is None:
        st.warning("Please generate or load a graph first.")
    else:
        removed_nodes = list(st.session_state.graph.removed_nodes.keys())
        node_to_restore = st.selectbox("Select Node to Restore", removed_nodes, index=0 if removed_nodes else 0)
        if st.button("Restore Node"):
            start_time = time.time()
            st.session_state.graph.restore_node(node_to_restore)
            st.success(f"Node {node_to_restore} restored in {time.time() - start_time:.2f} seconds!")
            st.write(f"Updated Graph: {st.session_state.graph}")
            health = st.session_state.graph.network_health_status()
            st.write("**Updated Health Status:**")
            st.json(health)

if option == "Restore Edge":
    st.header("Restore Edge")
    if st.session_state.graph is None:
        st.warning("Please generate or load a graph first.")
    else:
        removed_edges = list(st.session_state.graph.removed_edges.keys())
        edge_to_restore = st.selectbox("Select Edge to Restore", removed_edges, index=0 if removed_edges else 0)
        if st.button("Restore Edge"):
            start_time = time.time()
            st.session_state.graph.restore_edge(edge_to_restore[0], edge_to_restore[1])
            st.success(f"Edge {edge_to_restore[0]} -> {edge_to_restore[1]} restored in {time.time() - start_time:.2f} seconds!")
            st.write(f"Updated Graph: {st.session_state.graph}")
            health = st.session_state.graph.network_health_status()
            st.write("**Updated Health Status:**")
            st.json(health)