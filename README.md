## Environment

```
$ python3 -m venv linear_regression
$ source ./linear_regression/bin/activate
```


The first program will be used to predict the price of a car for a given mileage.
When you launch the program, it should prompt you for a mileage, and then give you back the estimated price for that mileage. The program will use the following hypothesis to predict the price : estimate Price(mileage) = θ0 + (θ1 ∗ mileage)
Before the run of the training program, theta0 and theta1 will be set to 0.

---
A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

Machine learning is a growing field of computer science that may seem a bit complicated and reserved only to mathematicians. You may have heard of neural networks or k-means clustering and don’t undersdand how they work or how to code these kinds of algorithms...
But don’t worry, we are actually going to start with a simple, basic machine learning algorithm.

The aim of this project is to introduce you to the basic concept behind machine learning.
For this project, you will have to create a program that predicts the price of a car by using a linear function train with a gradient descent algorithm.
We will work on a precise example for the project, but once you’re done you will be able to use the algorithm with any other dataset.

In this project you are free to use whatever language you want.
You are also free to use any libraries you want as long as they do not do all the work for you. For example, the use of python’s numpy.polyfit is considered cheating.

You should use a language that allows you to easily visualize your data : it will be very helpful for debugging.

You will implement a simple linear regression with a single feature - in this case, the mileage of the car.
To do so, you need to create two programs :

The first program will be used to predict the price of a car for a given mileage.
When you launch the program, it should prompt you for a mileage, and then give you back the estimated price for that mileage. The program will use the following hypothesis to predict the price : estimateP rice(mileage) = θ0 + (θ1 ∗ mileage)
Before the run of the training program, theta0 and theta1 will be set to 0.

The second program will be used to train your model. It will read your dataset file and perform a linear regression on the data.
Once the linear regression has completed, you will save the variables theta0 and theta1 for use in the first program.
You will be using the following formulas :

I let you guess what m is :)
Note that the estimatePrice is the same as in our first program, but here it uses your temporary, lastly computed theta0 and theta1.
Also, don’t forget to simultaneously update theta0 and theta1.

Here are some bonuses that could be very useful :
• Plotting the data into a graph to see their repartition.
• Plotting the line resulting from your linear regression into the same graph, to see the result of your hard work !
• A program that calculates the precision of your algorithm.
... and any other bonuses that make your program more awesome.

Your program will be reviewed by humans only, so you are free to organize your files whatever way you want.
Here are the points that your peer-corrector will have to check :
• The absence of libraries that do the work for you
• The use of the specified hypothesis
• The use of the specified training function