import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
from datetime import datetime
import os

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.001 # 0.00001
        self.initial_epsilon = 0.01
        self.number_of_iterations = 220000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out


def preprocess_frame(obs, device):
    frame = obs.reshape(12, 15)
    resized = cv2.resize(frame, (84, 84))
    normalized = resized.astype(np.float32)
    return torch.FloatTensor(normalized).unsqueeze(0).to(device)


def test_model(model, env, device, num_episodes=1):
    model.eval()
    test_scores = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_score = 0

        processed_frame = preprocess_frame(obs, device)
        state = torch.cat([processed_frame for _ in range(4)]).unsqueeze(0)

        while not done:
            env.render()

            with torch.no_grad():
                output = model(state)[0]
                action = torch.argmax(output).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_score += reward

            if not done:
                processed_obs = preprocess_frame(obs, device)
                state = torch.cat((state.squeeze(0)[1:], processed_obs)).unsqueeze(0)

        test_scores.append(episode_score)
        print(f"Test Episode {episode + 1}: Score = {episode_score}")

    return test_scores

def save_checkpoint(model, optimizer, episode, score, epsilon, save_dir, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'score': score,
        'epsilon': epsilon,
    }
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training environment
    env = gym.make('FlappyBird-v0', render_mode="rgb_array")
    model = NeuralNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.MSELoss()

    replay_memory = deque(maxlen=model.replay_memory_size)
    epsilon = model.initial_epsilon
    epsilon_decrements = np.linspace(model.initial_epsilon,
                                     model.final_epsilon,
                                     model.number_of_iterations)

    scores = []
    current_score = 0
    episode = 0

    obs, info = env.reset()
    processed_frame = preprocess_frame(obs, device)
    state = torch.cat([processed_frame for _ in range(4)]).unsqueeze(0)

    # Training loop
    iteration = 0
    while iteration < model.number_of_iterations:
        output = model(state)[0]

        if random.random() <= epsilon:
            action = random.randint(0, 1)
        else:
            action = torch.argmax(output).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        current_score += reward

        processed_obs = preprocess_frame(obs, device)
        next_state = torch.cat((state.squeeze(0)[1:], processed_obs)).unsqueeze(0)

        replay_memory.append((state.cpu(), action, reward, next_state.cpu(), done))

        if len(replay_memory) >= model.minibatch_size:
            minibatch = random.sample(replay_memory, model.minibatch_size)

            state_batch = torch.cat([d[0] for d in minibatch]).to(device)
            action_batch = torch.LongTensor([d[1] for d in minibatch]).to(device)
            reward_batch = torch.FloatTensor([d[2] for d in minibatch]).to(device)
            next_state_batch = torch.cat([d[3] for d in minibatch]).to(device)
            done_batch = torch.FloatTensor([d[4] for d in minibatch]).to(device)

            next_q_values = model(next_state_batch).max(1)[0].detach()
            target_q_values = reward_batch + (1 - done_batch) * model.gamma * next_q_values
            current_q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

            loss = criterion(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        epsilon = epsilon_decrements[iteration]

        if done:
            episode += 1
            scores.append(current_score)
            avg_score = np.mean(scores[-100:])
            print(f"Episode: {episode}, Score: {current_score}, Avg Score: {avg_score:.2f}, Epsilon: {epsilon:.4f}")

            obs, info = env.reset()
            processed_frame = preprocess_frame(obs, device)
            state = torch.cat([processed_frame for _ in range(4)]).unsqueeze(0)
            current_score = 0

        iteration += 1

    env.close()

    # Save the final model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_avg_score = np.mean(scores[-100:])
    save_name = f"model_ep_{episode}_avg_{final_avg_score:.2f}_{timestamp}.pth"
    save_path = os.path.join('saved_models', save_name)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'final_avg_score': final_avg_score,
        'epsilon': epsilon,
        'scores': scores
    }
    torch.save(checkpoint, save_path)
    print(f"\nModel saved as: {save_path}")

    # Testing phase
    print("\nTraining completed. Testing the model...")
    test_env = gym.make('FlappyBird-v0', render_mode="human")
    test_scores = test_model(model, test_env, device, num_episodes=3)
    print(f"\nAverage Test Score: {np.mean(test_scores):.2f}")
    test_env.close()


def load_and_test_model(
        model_path,
        num_episodes=3,
        render_mode="human",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model and load checkpoint
    model = NeuralNetwork().to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}

    env = gym.make('FlappyBird-v0', render_mode=render_mode)

    test_scores = test_model(model, env, device, num_episodes)

    results = {
        'mean_score': np.mean(test_scores),
        'num_episodes': num_episodes,
    }

    print("\nTest Results:")
    print(f"Number of episodes: {results['num_episodes']}")
    print(f"Average score: {results['mean_score']:.2f}")

    env.close()
    return results

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    model_path = "saved_models/model_ep_928_avg_41.82_20250112_152812.pth"
    results = load_and_test_model(
        model_path=model_path,
        num_episodes=1,
        render_mode="human",
    )