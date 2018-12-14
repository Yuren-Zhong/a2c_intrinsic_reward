# A2C with intrinsic reward

### Prerequisite 
The following packages should be installed:
```
torch
numpy
gym
gym[atari]
opencv
```
Additionally, `CUDA` should be available.
### To run the agents
```bash
# For simple policy-gradient:
python pg.py
# For simple policy-gradient with intrinsic reward:
python pg_r_in.py
# For A2C:
python original_actor_critic.py
# For A2C with intrinsic reward:
python intrinsic_reward_actor_critic.py
```
