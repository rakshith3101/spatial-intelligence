# Example Web Sources for fetch_and_walk.py
# Copy and paste URLs into the command line when running fetch_and_walk.py

## Example 1: Philosophy & Science
python fetch_and_walk.py \
  --urls https://en.wikipedia.org/wiki/Metaphysics \
         https://en.wikipedia.org/wiki/Quantum_mechanics \
  --iterations 8 \
  --walk-steps 5 \
  --threshold 0.42 \
  --visualize

## Example 2: Art & Mathematics
python fetch_and_walk.py \
  --urls https://en.wikipedia.org/wiki/Symmetry_in_art \
         https://en.wikipedia.org/wiki/Topology \
  --iterations 8 \
  --walk-steps 4 \
  --visualize

## Example 3: History & Technology
python fetch_and_walk.py \
  --urls https://en.wikipedia.org/wiki/History_of_computing \
         https://en.wikipedia.org/wiki/Ancient_Rome \
  --iterations 10 \
  --walk-steps 5 \
  --visualize

## Example 4: Multiple Technical Topics
python fetch_and_walk.py \
  --urls https://en.wikipedia.org/wiki/Machine_learning \
         https://en.wikipedia.org/wiki/Neural_network \
         https://en.wikipedia.org/wiki/Artificial_intelligence \
  --iterations 10 \
  --walk-steps 5 \
  --threshold 0.40 \
  --visualize

## Notes:
# - URLs should be accessible from your network
# - Processing time depends on page size and iteration count
# - Generated visualizations appear in visualizations/ directory
# - Results are saved to data/processed/
# - Adjust --threshold lower (0.35-0.40) for more connections
# - Increase --iterations for more hypothesis generation
# - Increase --walk-steps for longer semantic paths

## Popular Test Sources:
# - Wikipedia articles (most reliable)
# - Academic papers (if accessible)
# - Blog posts with long-form content
# - Documentation pages

## Tips for Interesting Results:
# 1. Mix domains that seem unrelated (philosophy + astronomy)
# 2. Use detailed, concept-rich content
# 3. Try 8-10 iterations for good discovery
# 4. Lower threshold (0.38-0.42) for more creative chains
# 5. Combine 2-3 URLs for cross-domain insights
