# Flappy Bird with Deep Reinforcement Learning
Flappy Bird Game trained on 3 models DQN algorithm: Deep Q-Network, Double DQN (DDQN) and Dueling DDQN with Prioritized Experience Replay implemented using Pytorch.

### Prerequisites
You will need Python 3.x.x with some packages which you can install direclty using requirements.txt.
> pip install -r requirements.txt

### Running The Game
Use the following command to run the game where '--model' indicates the location of saved DQN model.
> python3 play_game.py --model checkpoints/best_DQN.dat
