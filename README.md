# Super-Mario-Bros Reinforcement Learning: QL vs Sarsa
The following project concerns the development of an intelligent agent for the famous game produced by Nintendo Super Mario Bros. More in detail: the goal of this project was to design, implement and train   an agent with the Q-learning reinforcement learning algorithm. Subsequently, the results of learning with the Q-learning algorithm were compared with the SARSA algorithm. In our case study, other types of    learning involving the double Q-learning algorithm, Deep Q-Network (DQN) and Double Deep Q-Network (DDQN). The reason why these different learnings are provided is for performance issues. For more information, read the <a href="https://github.com/AlessandroUnisa">report</a> written by us.

The parameters and plots of the relevant QL models are located under `./code/Reinforcement_Learning/models`, while the parameters and plots of the Sarsa models are located under `./code/Reinforcement_Learning/sarsa/models`. 

![world-1-1-n_stack=4](https://github.com/Alberto-00/Super-Mario-Bros-AI/blob/main/demo/ddqn.gif)

## Requirements (tested)

| Module               | Version |
|----------------------|---------|
| gym                  | 0.25.2  |
| gym-super-mario-bros | 7.4.0   |
| nes-py               | 8.2.1   |
| pyglet               | 1.5.21  |
| torch                | 2.1.1   |
| pygame               | 2.5.2   |


## Gym Environment
I used the [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) environment. The code can be found in `./code/Reinforcement_Learning/utils/enviroment.py`, where I do the setup of the environment. In `./code/Reinforcement_Learning/utils/setup_env.py` I assign custom values to the rewards so as to take as many power-ups as possible. Then the agents QL logic can be found in `./code/Reinforcement_Learning/utils/agents`, while models and Sarsa agents can be found in `./code/Reinforcement_Learning/sarsa`
The custom rewards are:

*	<i>time</i>: -0.1,  per second that passes
* <i>death</i>: -100.,  mario dies
* <i>extra_life</i>: 100.,   mario gets an extra life
* <i>mushroom</i>: 20.,   mario eats a mushroom to become big
* <i>flower</i>: 25.,   mario eats a flower
* <i>mushroom_hit</i>: -10.,   mario gets hit while big
* <i>flower_hit</i>: -15.,   mario gets hit while fire mario
* <i>coin</i>: 15.,   mario gets a coin
* <i>score</i>: 15.,   mario hit enemies
* <i>victory</i>: 1000   mario win

## Training & Results

I used the QL, Double QL, Deep QN, Double Deep QN agents together with their respective sarsa agents with epsilon-greedy policy. Each model was trained for 1000 steps and took about 3.5 hours to finish except for DDQN and DDN Sarsa that they was trained for 10.000 steps and took about 13.4 hours.

Here are the results of all the models, specifically we make a comparison between the QL and Sarsa algorithms.

<img src="https://github.com/Alberto-00/Super-Mario-Bros-AI/blob/main/grafici/ql_vs_sarsa.png" width="500"> <img src="https://github.com/Alberto-00/Super-Mario-Bros-AI/blob/main/grafici/doubleQL_vs_doubleSarsa.png" width="500" >
<img src="https://github.com/Alberto-00/Super-Mario-Bros-AI/blob/main/grafici/dqn_vs_dnSarsa.png" width="500" > <img src="https://github.com/Alberto-00/Super-Mario-Bros-AI/blob/main/grafici/ddqn_vs_ddnSarsa.png" width="500" >





|                  |![world-1-1-n_stack=1](https://github.com/Alberto-00/Super-Mario-Bros-AI/blob/main/demo/ddn_sarsa_lose.gif)|![world-1-1-n_stack=2](https://github.com/Alberto-00/Super-Mario-Bros-AI/blob/main/demo/ddn_sarsa_victory.gif)|![world-1-1-n_stack=4](https://github.com/Alberto-00/Super-Mario-Bros-AI/blob/main/demo/ddqn.gif)|
|------------------|-------|------|------|
| Training steps   | 10K   | 10K  | 10K  |
| Episode score    | 1723  | 4100 | 4320 |
| Agents           | DDN Sarsa  | DDN Sarsa | DDQN |
| Completed level? | False | True | True |

So, to get more results, we could implement the PPO agorithm in both QL and Sarsa algorithms and make further comparisons in order to figure out which algorithm is best for the super mario bros game.

<h2 dir="auto">
  <a id="user-content-authors" class="anchor" aria-hidden="true" href="#authors">
    <svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z">
      </path>
    </svg>
  </a>
  Author & Contacts
</h2>


| Name | Description |
| --- | --- |
| <p dir="auto"><strong>Alberto Montefusco</strong> |<br>Developer - <a href="https://github.com/Alberto-00">Alberto-00</a></p><p dir="auto">Email - <a href="mailto:a.montefusco28@studenti.unisa.it">a.montefusco28@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alberto-montefusco">Alberto Montefusco</a></p><p dir="auto">My WebSite - <a href="https://alberto-00.github.io/">alberto-00.github.io</a></p><br>|
| <p dir="auto"><strong>Alessandro Aquino</strong> |<br>Developer   - <a href="https://github.com/AlessandroUnisa">AlessandroUnisa</a></p><p dir="auto">Email - <a href="mailto:a.aquino33@studenti.unisa.it">a.aquino33@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alessandro-aquino-62b74218a/">Alessandro Aquino</a></p><br>|
| <p dir="auto"><strong>Mattia d'Argenio</strong> |<br>Developer   - <a href="https://github.com/mattiadarg">mattiadarg</a></p><p dir="auto">Email - <a href="mailto:m.dargenio5@studenti.unisa.it">m.dargenio5@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/mattia-d-argenio-a57849255/)https://www.linkedin.com/in/mattia-d-argenio-a57849255/">Mattia d'Argenio</a></p><br>|
