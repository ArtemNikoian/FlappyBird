import numpy as np
import pickle
from env import FlappyBirdEnv

class NeuralNetwork:
    def __init__(self, input_size=4, hidden_size=8, output_size=2):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.random.randn(hidden_size) * 0.5
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.random.randn(output_size) * 0.5
    
    def forward(self, x):
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        out = np.dot(h, self.w2) + self.b2
        return out
    
    def get_action(self, obs):
        out = self.forward(obs)
        return np.argmax(out)
    
    def get_params(self):
        return [self.w1, self.b1, self.w2, self.b2]
    
    def set_params(self, params):
        self.w1, self.b1, self.w2, self.b2 = params

def evaluate_network(network, n_episodes=16):
    env = FlappyBirdEnv()
    total_reward = 0
    
    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = network.get_action(obs)
            obs, reward, done = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / n_episodes

def crossover(parent1, parent2):
    child = NeuralNetwork()
    params1 = parent1.get_params()
    params2 = parent2.get_params()
    child_params = []
    
    for p1, p2 in zip(params1, params2):
        mask = np.random.rand(*p1.shape) > 0.5
        child_param = np.where(mask, p1, p2)
        child_params.append(child_param)
    
    child.set_params(child_params)
    return child

def mutate(network, mutation_rate=0.1, mutation_scale=0.3):
    params = network.get_params()
    mutated_params = []
    
    for param in params:
        mask = np.random.rand(*param.shape) < mutation_rate
        noise = np.random.randn(*param.shape) * mutation_scale
        mutated_param = param + mask * noise
        mutated_params.append(mutated_param)
    
    network.set_params(mutated_params)
    return network

def genetic_algorithm(pop_size=50, generations=40, elite_size=5):
    # Initialize population
    population = [NeuralNetwork() for _ in range(pop_size)]
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_network(net) for net in population]
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        best_fitness = fitness_scores[0]
        avg_fitness = np.mean(fitness_scores)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        print(f"Generation {gen+1}/{generations} - Best: {best_fitness:.2f}, Avg: {avg_fitness:.2f}")
        
        # Keep elite
        new_population = population[:elite_size]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            tournament_size = 5
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            parent1 = population[tournament[np.argmax([fitness_scores[i] for i in tournament])]]
            
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            parent2 = population[tournament[np.argmax([fitness_scores[i] for i in tournament])]]
            
            # Crossover and mutation
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    # Save best network
    best_network = population[0]
    with open('best_flappy_bird.pkl', 'wb') as f:
        pickle.dump(best_network, f)
    
    print(f"\nTraining complete! Best fitness: {best_fitness_history[-1]:.2f}")
    print("Model saved as 'best_flappy_bird.pkl'")
    
    return best_network, best_fitness_history, avg_fitness_history

if __name__ == "__main__":
    best_net, best_hist, avg_hist = genetic_algorithm(pop_size=50, generations=30)