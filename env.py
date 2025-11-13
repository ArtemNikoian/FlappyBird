import numpy as np

class FlappyBirdEnv:
    def __init__(self):
        self.screen_width = 400
        self.screen_height = 600
        self.bird_x = 50
        self.pipe_width = 60
        self.gap_size = 150
        self.pipe_spacing = 300
        self.gravity = 0.6
        self.flap_strength = -10
        
        self.reset()
    
    def reset(self):
        self.bird_y = self.screen_height // 2
        self.bird_vel = 0
        self.pipes = [self.screen_width + i * self.pipe_spacing for i in range(3)]
        self.pipe_heights = [np.random.randint(100, self.screen_height - self.gap_size - 100) 
                            for _ in range(3)]
        self.score = 0
        self.frames = 0
        self.passed_pipes = set()
        return self._get_observation()
    
    def _get_observation(self):
        # Find next pipe (closest one that hasn't been fully passed)
        next_pipe_idx = None
        min_x = float('inf')
        
        for i, pipe_x in enumerate(self.pipes):
            if pipe_x + self.pipe_width > self.bird_x:  # Pipe hasn't been fully passed
                if pipe_x < min_x:  # Find the closest one
                    min_x = pipe_x
                    next_pipe_idx = i
        
        if next_pipe_idx is None:
            # Fallback (shouldn't happen with proper spacing)
            next_pipe_idx = 0
        
        next_pipe_x = self.pipes[next_pipe_idx]
        next_pipe_hole_center = self.pipe_heights[next_pipe_idx] + self.gap_size / 2
        
        # Horizontal distance to beginning of next pipe
        horizontal_dist = (next_pipe_x - self.bird_x) / self.screen_width
        
        # Vertical displacement from hole center to bird
        vertical_dist = (self.bird_y - next_pipe_hole_center) / self.screen_height
        
        # Normalize velocity (typical range is roughly -15 to 15)
        velocity = self.bird_vel / 15.0
        
        # Boolean: inside pipe currently (horizontally)
        inside_pipe = 1.0 if (next_pipe_x <= self.bird_x <= next_pipe_x + self.pipe_width) else 0.0
        
        return np.array([horizontal_dist, vertical_dist, velocity, inside_pipe])
    
    def step(self, action):
        # action: 0 = do nothing, 1 = flap
        if action == 1:
            self.bird_vel = self.flap_strength
        
        # Apply gravity
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel
        
        # Initialize reward
        reward = 0
        
        # Move pipes
        for i in range(len(self.pipes)):
            self.pipes[i] -= 3
            
            # Recycle pipe
            if self.pipes[i] < -self.pipe_width:
                self.pipes[i] = max(self.pipes) + self.pipe_spacing
                self.pipe_heights[i] = np.random.randint(100, self.screen_height - self.gap_size - 100)
                if i in self.passed_pipes:
                    self.passed_pipes.remove(i)
            
            # Check if passed pipe
            if self.pipes[i] + self.pipe_width < self.bird_x and i not in self.passed_pipes:
                self.passed_pipes.add(i)
                self.score += 1
                reward += 1  # +1 for passing pipe
        
        self.frames += 1
        
        # Check collision
        done = False
        if self.bird_y < 0 or self.bird_y > self.screen_height:
            done = True
        
        # Check if max score reached
        if self.score >= 500:
            done = True
        
        # Check pipe collision
        for i, pipe_x in enumerate(self.pipes):
            if pipe_x <= self.bird_x <= pipe_x + self.pipe_width:
                pipe_top = self.pipe_heights[i]
                pipe_bottom = pipe_top + self.gap_size
                if self.bird_y < pipe_top or self.bird_y > pipe_bottom:
                    done = True
        
        return self._get_observation(), reward, done
    
    def get_state(self):
        # For rendering
        return {
            'bird_x': self.bird_x,
            'bird_y': self.bird_y,
            'pipes': self.pipes,
            'pipe_heights': self.pipe_heights,
            'score': self.score
        }