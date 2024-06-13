
###本工程文件适用于系统辨识小组作业
# Flappy Bird with Deep Reinforcement Learning
Flappy Bird Game trained on 3 models DQN algorithm: Deep Q-Network, Double DQN (DDQN) and Dueling DDQN with Prioritized Experience Replay implemented using Pytorch.


### Prerequisites
You will need Python 3.x.x with some packages which you can install direclty using requirements.txt.
> pip install -r requirements.txt

### Running The Game(切换Dueling_DQN时，需要先将play中的net换成DDQN）
first ,change the network form in paly_game.py(DQN or DDQN) 
THEN  ,Use the following command to run the game where '--model' indicates the location of saved DQN model.
> python play_game.py --model checkpoints/new/best_DQN.dat
>
### see the demo in 'video' directory(demo视频在video文件夹）


### see the eventout file in sclar/scalar2/scalar3,corrisponding to the DQN,DDQN,Dueling_DQN(tensorboard记录数据文件在scalar文件夹）
