---
title: 'chessvision-py: A Python Package for Computational Chess Performance Analysis'
tags:
  - Python
  - R
  - chess
  - machine learning
  - behavioral analysis
  - natural language processing
  - LSTM
authors:
  - name: Rakkshet Singhaal
    orcid: 0009-0007-2150-9876
    affiliation: 1
affiliations:
  - name: Kellogg School of Management, Northwestern University
    index: 1
date: 12 April 2026
bibliography: paper.bib
---

# Summary

`chessvision-py` is an open-source Python library that transforms personal chess game archives in Portable Game Notation (PGN) format into structured longitudinal datasets and applies machine learning to produce actionable, causally grounded performance insights. The package implements a five-stage pipeline: PGN parsing, Stockfish position evaluation, behavioral feature engineering, unsupervised error archetype discovery, and personalized training recommendation generation. An accompanying R package (`chessvision`) provides a fully documented interface to the Python backend via `reticulate` [@ushey2023reticulate], and an interactive Shiny dashboard enables non-programmatic access for coaches and practitioners.

# Statement of Need

Chess improvement is poorly understood at the individual level. Aggregate statistics such as mean centipawn loss (CPL) and blunder rates are widely reported by commercial platforms, but they conflate qualitatively distinct error types and offer no causal mechanism linking behavioral patterns to rating trajectories. Coaching literature identifies specific skill domains — endgame technique, tactical vision, positional understanding — as the primary levers of improvement, yet no open-source tool exists that (1) links these domains to empirically identified error clusters, (2) embeds individual playing style in a population-level vector space, or (3) generates causally grounded training recommendations from panel data on game performance.

`chessvision-py` fills this gap. It is designed for three audiences: amateur players seeking structured self-improvement guidance; coaches requiring systematic comparison tools across student portfolios; and researchers studying skill acquisition, decision-making under time pressure, and behavioral economics of competitive games.

# Implementation

## PGN Parsing and Evaluation

The `parse_pgn()` function reads PGN files from chess.com, lichess.org, and over-the-board tournament exports, producing two tidy `pandas` DataFrames: one row per game and one row per move. The `evaluate_games()` function evaluates every position using Stockfish [@stockfish] at configurable depth, caching results in a local SQLite database to support resumable evaluation of large archives. From each position evaluation, the package computes centipawn loss (CPL), phase labels (opening, middlegame, endgame), and error classification (blunder ≥300 CPL, mistake ≥100 CPL, inaccuracy ≥50 CPL).

## Behavioral Feature Engineering

The `engineer_features()` function computes five behavioral features per move: clock pressure index (time remaining normalized by initial time control), position complexity score (number of legal moves divided by mean legal moves in the game), novelty move index (binary indicator for moves leaving known opening theory), mobility ratio (player legal moves divided by opponent legal moves), and opening family code (ECO classification). These features form the input to downstream ML models.

## Error Archetype Discovery

The `run_archetype_analysis()` function applies a two-stage unsupervised clustering pipeline to error moves (CPL ≥ 50). First, UMAP [@mcinnes2018umap] reduces the behavioral feature matrix to 10 dimensions, preserving local and global structure. Second, HDBSCAN [@campello2013density] clusters the reduced representation into behaviorally distinct error archetypes without requiring a pre-specified number of clusters. Applied to 96,651 error moves from 4,685 games, the pipeline recovers five named archetypes: positional confusion (57.9%), endgame failure (19.0%), preparation boundary (15.4%), strategic drift (3.5%), and tactical blindspot (2.7%).

## Style Embedding

A Word2Vec skip-gram model [@mikolov2013efficient] — termed chess2vec — is trained on 2.44 billion move tokens derived from 29.3 million games in the Lichess Elite Database [@lichess], comprising games between players rated 2,200 and above. Each move is tokenized as a structured string encoding the algebraic notation, game phase, player color, position evaluation bucket, and clock pressure bucket. The resulting 128-dimensional embeddings capture stylistic co-occurrence patterns: the cosine similarity between Tal and Nezhmetdinov (both attacking players) is 0.979, while the cross-cluster similarity between attacking and positional players averages 0.852 — a separation of 0.115 cosine units recovered without labeled training data.

## ELO Trajectory Forecasting

An LSTM [@hochreiter1997long] model is trained on game trajectories from 67,115 elite players extracted from the Lichess Elite Database, achieving 53.2% direction accuracy on held-out players — above the 50% random baseline. The model accepts sequences of game-level behavioral features and predicts ELO change over the subsequent 10 games.

## Personalized Recommendation

The `analyze_player()` function combines error archetype profile, chess2vec style vector, and ELO trajectory to generate a ranked training plan. Recommendations are stratified by error frequency and matched to evidence-based instructional resources, with opening suggestions adapted to the player's style archetype.

# Pre-trained Models

Pre-trained model weights are hosted on Hugging Face Hub [@wolf2020huggingfaces] at `rakkshet/chessvision-models` and downloaded automatically on first use via `cv.download_models()`. The chess2vec model (180MB) captures population-level move co-occurrence patterns; the population LSTM (5MB) encodes ELO trajectory dynamics from 67,115 players; and the HDBSCAN error archetype model (50MB) clusters error moves into interpretable behavioral categories.

# Availability

`chessvision-py` is available on PyPI and GitHub under the GPL-3.0 license. The companion R package `chessvision` is available on GitHub and submitted to CRAN. An interactive Shiny dashboard is available in the `chessvision-app` repository. All code, data pipelines, and pre-trained models are fully reproducible from the provided scripts.

# Acknowledgements

The author thanks the Lichess.org team for maintaining the open Lichess Elite Database and the Stockfish development team for the open-source chess engine.

# References
