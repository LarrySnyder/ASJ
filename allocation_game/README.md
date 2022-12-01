# The Mobile Food Bank Game

"The Mobile Food Bank Game" was developed by Oumaima Sohab based on the problem tackled by [Sinclair et al](https://arxiv.org/abs/2105.05308). The purpose of the game is to show the students the trade-off between efficiency (minimizing waste) and fairness in the context of a sequential allocation problem.

Oumaima was the TA for the [Algorithms and Social Justice](https://github.com/LarrySnyder/ASJ) class taught by Larry Snyder (ISE) and Suzanne Edwards (English, WGSS) at Lehigh University in Fall 2022 and developed this game to play in class.

![game running gif](images/game.gif)

## 🧐 About the game

Here's the (fictional) setup to the game: Due to the pandemic, the Lehigh food bank set up a mobile pantry that travels to 8 different locations in the Lehigh Valley, PA. You play as the operator of the mobile pantry. At each location, you need to choose how much food (meat, pasta, and prepared meals) to give to people who show up. They can be classified based on their dietary preferences to omnivores, vegetarians, and meal-prep. For the sake of simplicity, omnivores only eat meat, vegetarians only eat pasta, and meal-prep only eat prepared meals. Therefore the amount of meat chosen will be  equally distributed and given the omnivores , and so on. Since you have limited ressources, once you leave a location (hit "Next" in the game), there is no going back and the food left at the end goes to waste, so make your choices wisely!

In the first round, you are only evaluated based on the amount of food left at the end of the game. However, the second round includes a fairness score as well as satisfaction level (using <img src="images/100.png" width="20">,  <img src="images/80.png" width="20">, <img src="images/60.png" width="20">,  <img src="images/40.png" width="20"> and <img src="images/20.png" width="20">) of the visitors of each location. 

## 👨‍💻 Teaching using the game

We suggest the following plan:

1. Introduce the game and its rules
2. Have the students play the first round (no fairness)
3. Brainstorm fairness measures and introduce *envy-freeness* (the fairness metric used by Sinclair et al.)
4. Have the students play the second round and save their scores
5. Explain the optimization problem "solved" in each round
6. Plot efficiency x-axis vs fairness y-axis scores to show the trade-off between the two


## ⚠️ Requirements

* Python >= 3.8
* Numpy
* Pygame
