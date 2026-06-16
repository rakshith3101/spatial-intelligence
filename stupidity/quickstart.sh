#!/bin/bash
# Quick Start - Semantic Mutation Engine

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  AI Spectra: Semantic Mutation Engine                          ║"
echo "║  Fast Start Guide                                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv venv
    
    echo "📥 Activating venv..."
    source venv/Scripts/activate 2>/dev/null || . venv/bin/activate
    
    echo "📚 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "✓ Virtual environment found"
    source venv/Scripts/activate 2>/dev/null || . venv/bin/activate
fi

echo ""
echo "🚀 Choose what to run:"
echo ""
echo "1) Discover Connections (Indian History + Aerospace)"
echo "   python explore_facts.py --visualize"
echo ""
echo "2) Extract & Analyze Web Content"
echo "   python fetch_and_walk.py --urls <url1> <url2> --visualize"
echo ""
echo "3) Run Original Pipeline"
echo "   python main.py --iterations 5"
echo ""
echo "4) View Current Graph"
echo "   python -c \"import networkx as nx; import pickle; g = pickle.load(open('data/graph/graph.gpickle', 'rb')); print(f'Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}') if g else print('No graph yet')\""
echo ""
echo "📖 Full documentation: see README.md"
echo ""
