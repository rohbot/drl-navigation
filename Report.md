# Report
---
This project implements a reinforcement learning agent to navigate in a large, square world filled with yellow and blue bananas. For every yellow banana the agent collects it recieves a reward of +1 and -1 for every blue banana. The objective is to navigate around the space to collect all the yellow bananas and discard the blue ones. The environment is considered solved when the agent achieves an average score of +13  of +13 over 100 consecutive episodes.

The agents is trained using a Deep Q-Network (DQN) architecture based on the [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

## Implementation
---

The implementation split into a few smaller modules: 

* model.py - Neural Network model implemented with PyTorch
* dqn_agent.py - DQN agent implementation as describe in [paper](https://www.nature.com/articles/nature14236) mentioned above
* Navigation.ipynb - imports all required modules and allows the enviroment to be explored and the agent trained

### DQN Model

  



## Results
---
The agent achieves a score of 13.01 after 163 episodes
![alt text](data/images/episodes.png "Training Episodes")

<a href="http://www.youtube.com/watch?feature=player_embedded&v=TcdwhNYr7Hc
" target="_blank"><img src="http://img.youtube.com/vi/TcdwhNYr7Hc/0.jpg" 
alt="DQN Banana Collector" width="240" height="180" border="10" /></a>



## Further Improvements
---
* Double Q 
* 
