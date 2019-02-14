[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


#Continuous Control

### Information
This project is for teaching a unity robot, which deals in continous action space to take actions such that it maximizes its probablity of reaching its arm towards the goal.

Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		goal_speed -> 1.0
		goal_size -> 5.0
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
        
### Installation

``` bash
conda install --file python3.yml
```

### File Info

Reacher.app --> Unity enviornment for Mac devices
checkpoint_actor.pth,checkpoint_critic.pth ----> Saved model
ddpg_agent.py --> agent file for taking actions and steps
model2.py --> models for actor and critic
run.py --> file for training
temp.ipynb --> Ipython notebok for training




