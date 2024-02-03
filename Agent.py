import numpy as np
import random

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import pyspiel


# Some helper classes and functions.
# DO NOT CHANGE.

class BotAgent(rl_agent.AbstractAgent):
  """Agent class that wraps a bot.

  Note, the environment must include the OpenSpiel state in its observations,
  which means it must have been created with use_full_state=True.

  This is a simple wrapper that lets the RPS bots be interpreted as agents under
  the RL API.
  """

  def __init__(self, num_actions, bot, name="bot_agent"):
    assert num_actions > 0
    self._bot = bot
    self._num_actions = num_actions

  def restart(self):
    self._bot.restart()

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return
    _, state = pyspiel.deserialize_game_and_state(
        time_step.observations["serialized_state"])
    action = self._bot.step(state)
    probs = np.zeros(self._num_actions)
    probs[action] = 1.0
    return rl_agent.StepOutput(action=action, probs=probs)



def get_reward(action, opponent_move):
        # Rock-Paper-Scissors rules:
        # Rock beats Scissors, Scissors beats Paper, Paper beats Rock
        if action == opponent_move:
            return 0  # Tie
        elif (
            (action == "rock" and opponent_move == "scissors") or
            (action == "scissors" and opponent_move == "paper") or
            (action == "paper" and opponent_move == "rock")
        ):
            return 1  # Win
        else:
            return -1  # Loss


#  We will use this function to evaluate the agents. Do not change.
def eval_agents(env, agents, num_players, num_episodes, verbose=False):
  """Evaluate the agent.

  Runs a number of episodes and returns the average returns for each agent as
  a numpy array.

  Arguments:
    env: the RL environment,
    agents: a list of agents (size 2),
    num_players: number of players in the game (for RRPS, this is 2),
    num_episodes: number of evaluation episodes to run.
    verbose: whether to print updates after each episode.
  """
  sum_episode_rewards = np.zeros(num_players)
  for ep in range(num_episodes):
    for agent in agents:
      # Bots need to be restarted at the start of the episode.
      if hasattr(agent, "restart"):
        agent.restart()
    time_step = env.reset()
    episode_rewards = np.zeros(num_players)
    while not time_step.last():
      agents_output = [
          agent.step(time_step, is_evaluation=True) for agent in agents
      ]
      action_list = [agent_output.action for agent_output in agents_output]
      time_step = env.step(action_list)
      episode_rewards += time_step.rewards
    sum_episode_rewards += episode_rewards
    if verbose:
      print(f"Finished episode {ep}, "
            + f"avg returns: {sum_episode_rewards / (ep+1)}")

  return sum_episode_rewards / num_episodes


def print_roshambo_bot_names_and_ids(roshambo_bot_names):
  print("Roshambo bot population:")
  for i in range(len(roshambo_bot_names)):
    print(f"{i}: {roshambo_bot_names[i]}")

def create_roshambo_bot_agent(player_id, num_actions, bot_names, pop_id):
  name = bot_names[pop_id]
  # Creates an OpenSpiel bot with the default number of throws
  # (pyspiel.ROSHAMBO_NUM_THROWS). To create one for a different number of
  # throws per episode, add the number as the third argument here.
  bot = pyspiel.make_roshambo_bot(player_id, name)
  return BotAgent(num_actions, bot, name=name)


# Some basic info and initialize the population

# print(pyspiel.ROSHAMBO_NUM_BOTS)    # 43 bots
# print(pyspiel.ROSHAMBO_NUM_THROWS)  # 1000 steps per episode

# The recall is how many of the most recent actions are presented to the RL
# agents as part of their observations. Note: this is just for the RL agents
# like DQN etc... every bot has access to the full history.
RECALL = 20

# The population of 43 bots. See the RRPS paper for high-level descriptions of
# what each bot does.

# print("Loading bot population...")
pop_size = pyspiel.ROSHAMBO_NUM_BOTS
# print(f"Population size: {pop_size}")
roshambo_bot_names = pyspiel.roshambo_bot_names()
roshambo_bot_names.sort()
# print_roshambo_bot_names_and_ids(roshambo_bot_names)

bot_id = 0
roshambo_bot_ids = {}
for name in roshambo_bot_names:
  roshambo_bot_ids[name] = bot_id
  bot_id += 1


# Example: create an RL environment, and two agents from the bot population and
# evaluate these two agents head-to-head.

# Note that the include_full_state variable has to be enabled because the
# BotAgent needs access to the full state.
env = rl_environment.Environment(
    "repeated_game(stage_game=matrix_rps(),num_repetitions=" +
    f"{pyspiel.ROSHAMBO_NUM_THROWS}," +
    f"recall={RECALL})",
    include_full_state=True)
num_players = 2
num_actions = env.action_spec()["num_actions"]
# Learning agents might need this:
# info_state_size = env.observation_spec()["info_state"][0]

# Create two bot agents
p0_pop_id = 0   # actr_lag2_decay
p1_pop_id = 1   # adddriftbot2
agents = [
    create_roshambo_bot_agent(0, num_actions, roshambo_bot_names, p0_pop_id),
    create_roshambo_bot_agent(1, num_actions, roshambo_bot_names, p1_pop_id)
]

# print("Starting eval run.")
# avg_eval_returns = eval_agents(env, agents, num_players, 10, verbose=True)

# print("Avg return ", avg_eval_returns)


# Template to get you started. Pick one of these to implement as your starting
# point.

# Template : Basic RL agent.
#
#
class MyAgent(rl_agent.AbstractAgent):
    """Agent class that learns to play RRPS.

    You fill this in to create your RRPS agent.

    See the superclass for more info: https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_agent.py
    """

    def __init__(self, num_actions, name="bot_agent"):
        assert num_actions > 0
        self._num_actions = num_actions  # 3
        self._alpha = 0.8  # Learning rate
        self._count = 0
        self._episode_num = 0
        self._gamma = 0.8  # Discount factor
        self._epsilon = 0.4  # Exploration-exploitation trade-off
        states = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2)
        ]
        self._q_table = dict()
        self._opponent_track = dict()
        self._rewards = {
            (0, 0): 0,
            (0, 1): -1,
            (0, 2): 1,
            (1, 0): 1,
            (1, 1): 0,
            (1, 2): -1,
            (2, 0): -1,
            (2, 1): 1,
            (2, 2): 0
        }
        for s in states:
            self._q_table[s] = {0: 0, 1: 0, 2: 0}
            self._opponent_track[s] = {0: 0, 1: 0, 2: 0}


    def choose_action(self, curstate):
        # return 0
        # Exploration-exploitation: choose a random action with probability epsilon
        if (random.uniform(0, 1) < self._epsilon):
            return random.choice(range(self._num_actions))
        else:
            # Choose the action with the highest Q-value for the current state
            return max(self._q_table.get(curstate, {}), key=self._q_table[curstate].get, default=0)
    

    def choose_action_next(self, curstate):
        # return 0
        # Exploration-exploitation: choose a random action with probability epsilon
        if (random.uniform(0, 1) < 0.1):
            return random.choice(range(self._num_actions))
        else:
            # Choose the action with the highest Q-value for the current state
            return max(self._q_table.get(curstate, {}), key=self._q_table[curstate].get, default=0)
            
    
    def update_q_table(self, current_state, action, reward, next_state, next_action):
        # Q-learning update rule
        current_q = self._q_table.get(current_state, {}).get(action, 0)
        max_future_q = self._q_table.get(next_state, {}).get(next_action, 0)
        new_q = current_q + self._alpha * (reward + self._gamma*max_future_q - current_q)
        self._q_table[current_state][action] = new_q
        # print(self._q_table)


    def step(self, time_step, is_evaluation=False):
        if (self._count % 1000 == 0 and self._alpha > 0.1):
            self._alpha -= 0.075
        if (self._count % 1000 == 0 and self._epsilon > 0.1):
            self._epsilon -= 0.05
        self._count += 1
        # If it is the end of the episode, don't select an action.
        if time_step.last():
            return
        # Note: If the environment was created with include_full_state=True, then
        # game and state can be obtained as follows:
        game, state = pyspiel.deserialize_game_and_state(time_step.observations["serialized_state"])
        # A useful piece of information is the history (previous actions taken by agents).
        # You can access this by state.history()

        # A simple state representation (you might want to customize this based on your needs)
        all_actions = state.history()

        if (len(all_actions) < 4):
            # Return the selected action and probabilities
            action = random.choice(range(self._num_actions))
            probs = np.zeros(self._num_actions)
            probs[action] = 1.0
            return rl_agent.StepOutput(action=action, probs=probs)
        
        current_state = (all_actions[-4], all_actions[-3])
        next_state = (all_actions[-2], all_actions[-1])
        # action = self.choose_action(current_state)
        # action = all_actions[-2]
        # print("CUR:", current_state)
        # print("NEXT:", next_state)
        # print("ACTION:", action)
        # print('\n')

        # Choose the next action based on the current state
        # reward = self._rewards.get(next_state)
        reward = self._rewards.get(current_state)
        # print("REWARD:", reward)
        self.update_q_table(current_state, all_actions[-2], reward, next_state, self.choose_action(next_state))

        # Return the selected action and probabilities
        action = self.choose_action(next_state)
        probs = np.zeros(self._num_actions)
        probs[action] = 1.0
        return rl_agent.StepOutput(action=action, probs=probs)

# Just trying an example out.

my_agent = MyAgent(3, name="sanat_agent")
# print(my_agent._num_actions)

p1_pop_id = 0
win = 0
total = 43
while (p1_pop_id < total):
    agents = [
        my_agent,
        create_roshambo_bot_agent(1, num_actions, roshambo_bot_names, p1_pop_id)
    ]

    print(f"Starting eval run. {p1_pop_id+1} {roshambo_bot_names[p1_pop_id]}")
    avg_eval_returns = eval_agents(env, agents, num_players, 10, verbose=False)

    print("Avg return ", avg_eval_returns)

    if avg_eval_returns[0] > avg_eval_returns[1]:
        win += 1

    p1_pop_id += 1

    print(f"Win Percentage: {(win/p1_pop_id)*100} %")
