# Flappy Bird AI

![Trained Agent Playing Flappy Bird](trained_agent.gif)

A Flappy Bird game implementation trained using genetic algorithms and neural networks.

## Overview

This project implements a Flappy Bird environment from scratch and trains an AI agent to play the game autonomously using evolutionary strategies (genetic algorithms). The AI learns to navigate through pipes by deciding when to flap its wings based on game state observations.

## Project Structure

- **env.py** - Game environment implementation
  - `FlappyBirdEnv`: The Flappy Bird game engine with physics simulation
  - Features: gravity, collision detection, pipe generation, score tracking
  - State space: 4-dimensional observations (horizontal distance to next pipe, vertical distance to pipe center, velocity, inside pipe flag)

- **train.py** - Training logic
  - `NeuralNetwork`: 2-layer neural network for decision making (4→8→2)
  - Genetic algorithm implementation with tournament selection, crossover, and mutation
  - Population size: 50, Generations: 30 (configurable)
  - Saved model: `best_flappy_bird.pkl`

- **test.py** - Testing and visualization
  - `test_model()`: Runs the trained agent with pygame visualization
  - `play_manual()`: Play the game manually with SPACE or mouse click

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install pygame numpy pillow imageio
```

## Usage

### Train the Model
```bash
python train.py
```
This will train a genetic algorithm for 30 generations and save the best network to `best_flappy_bird.pkl`.

### Test the Trained Agent
```bash
python test.py
```
This will run the trained agent for 5 episodes and display the results.

### Play Manually
```bash
python test.py manual
```
Press SPACE or click to make the bird flap.

## Technical Details

### Game Environment
- Screen size: 400×600 pixels
- Bird starting position: x=50, y=300
- Gravity: 0.6 pixels per frame²
- Flap strength: -10 pixels per frame
- Gap between pipes: 150 pixels
- Pipe width: 60 pixels
- Pipe spacing: 300 pixels

### Neural Network Architecture
- Input layer: 4 neurons (game state observations)
- Hidden layer: 8 neurons with tanh activation
- Output layer: 2 neurons (no flap / flap)
- Activation: argmax for action selection

### Genetic Algorithm Parameters
- Population size: 50
- Elite size: 5
- Mutation rate: 0.1
- Mutation scale: 0.3
- Tournament size: 5
- Generations: 30

## Results

The trained agent successfully learns to navigate through the pipes with consistent performance. The model achieves scores in the double digits on individual runs.

