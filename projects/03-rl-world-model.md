# Project 3 — RL + Latent World Model

## Goal
Learn a model of environment dynamics and use imagined rollouts for planning.

## Minimum pipeline
Environment → trajectories (s,a,r,s') → encoder → latent transition model → reward/value model → imagined rollout → action selection.

## Baselines
- model-free policy;
- one-step dynamics;
- multi-step latent rollout.

## Evaluation
Prediction error vs rollout horizon, return, planning compute, failure under model error.
