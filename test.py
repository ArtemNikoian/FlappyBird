import pygame
import pickle
import numpy as np
from env import FlappyBirdEnv
from train import NeuralNetwork

def play_manual(n_episodes=5):
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 600))
    pygame.display.set_caption('Flappy Bird - Manual Play')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    env = FlappyBirdEnv()
    
    total_scores = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = 0  # Default: do nothing
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = 1  # Flap
                if event.type == pygame.MOUSEBUTTONDOWN:
                    action = 1  # Flap
            
            obs, reward, done = env.step(action)
            episode_reward += reward
            
            # Render
            screen.fill((135, 206, 235))  # Sky blue
            
            # Draw pipes
            state = env.get_state()
            for i, pipe_x in enumerate(state['pipes']):
                pipe_top = state['pipe_heights'][i]
                pipe_bottom = pipe_top + env.gap_size
                
                # Top pipe
                pygame.draw.rect(screen, (0, 200, 0), 
                               (pipe_x, 0, env.pipe_width, pipe_top))
                # Bottom pipe
                pygame.draw.rect(screen, (0, 200, 0), 
                               (pipe_x, pipe_bottom, env.pipe_width, 
                                env.screen_height - pipe_bottom))
            
            # Draw bird
            pygame.draw.circle(screen, (255, 255, 0), 
                             (int(state['bird_x']), int(state['bird_y'])), 15)
            
            # Draw score
            score_text = font.render(f"Score: {state['score']}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            
            episode_text = font.render(f"Episode: {episode+1}/{n_episodes}", True, (255, 255, 255))
            screen.blit(episode_text, (10, 50))
            
            instruction_text = font.render("SPACE/CLICK to flap", True, (255, 255, 255))
            screen.blit(instruction_text, (10, 550))
            
            pygame.display.flip()
            clock.tick(60)
        
        total_scores.append(state['score'])
        print(f"Episode {episode+1} - Score: {state['score']}, Reward: {episode_reward:.2f}")
    
    pygame.quit()
    
    print(f"\nPerformance Summary:")
    print(f"Average Score: {np.mean(total_scores):.2f}")
    print(f"Best Score: {np.max(total_scores)}")
    print(f"Worst Score: {np.min(total_scores)}")

def test_model(model_path='best_flappy_bird.pkl', n_episodes=5):
    # Load model
    with open(model_path, 'rb') as f:
        network = pickle.load(f)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 600))
    pygame.display.set_caption('Flappy Bird AI')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    env = FlappyBirdEnv()
    
    total_scores = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Get action from network
            action = network.get_action(obs)
            obs, reward, done = env.step(action)
            episode_reward += reward
            
            # Render
            screen.fill((135, 206, 235))  # Sky blue
            
            # Draw pipes
            state = env.get_state()
            for i, pipe_x in enumerate(state['pipes']):
                pipe_top = state['pipe_heights'][i]
                pipe_bottom = pipe_top + env.gap_size
                
                # Top pipe
                pygame.draw.rect(screen, (0, 200, 0), 
                               (pipe_x, 0, env.pipe_width, pipe_top))
                # Bottom pipe
                pygame.draw.rect(screen, (0, 200, 0), 
                               (pipe_x, pipe_bottom, env.pipe_width, 
                                env.screen_height - pipe_bottom))
            
            # Draw bird
            pygame.draw.circle(screen, (255, 255, 0), 
                             (int(state['bird_x']), int(state['bird_y'])), 15)
            
            # Draw score
            score_text = font.render(f"Score: {state['score']}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            
            episode_text = font.render(f"Episode: {episode+1}/{n_episodes}", True, (255, 255, 255))
            screen.blit(episode_text, (10, 50))
            
            pygame.display.flip()
            clock.tick(60)
        
        total_scores.append(state['score'])
        print(f"Episode {episode+1} - Score: {state['score']}, Reward: {episode_reward:.2f}")
    
    pygame.quit()
    
    print(f"\nPerformance Summary:")
    print(f"Average Score: {np.mean(total_scores):.2f}")
    print(f"Best Score: {np.max(total_scores)}")
    print(f"Worst Score: {np.min(total_scores)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'manual':
        play_manual()
    else:
        test_model()