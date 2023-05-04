Download Link: https://assignmentchef.com/product/solved-cs771a-homework1
<br>
(Absolute Loss Regression with Sparsity) The absolute loss regression problem with <em>`</em><sub>1 </sub>regularization is

<em>N</em>

<em>w</em><em><sub>opt </sub></em>= argmin<sup>X</sup>|<em>y<sub>n </sub></em>− <em>w</em><sup>&gt;</sup><em>x</em><em><sub>n</sub></em>| + <em>λ</em>||<em>w</em>||<sub>1</sub>

<em>w</em>

<em>n</em>=1

where is the absolute value function, and <em>λ &gt; </em>0 is the regularization hyperparameter.

Is the above objective function convex? You don’t need to prove this formally; just a brief reasoning based on properties of other functions that are known to be convex/non-convex would be fine.

Derivate the expression for the (sub)gradient vector for this model.

<h1>Problem 2</h1>

(Feature Masking as Regularization) Consider linear regression model by minimizing the squared loss function. Suppose we decide to mask out or “drop” each feature <em>x<sub>nd </sub></em>of each input <em>x</em><em><sub>n </sub></em>∈ R<em><sup>D</sup></em>, independently, with probability 1 − <em>p </em>(equivalently, retaining the feature with probability <em>p</em>). Masking or dropping out basically means that we will set the feature <em>x<sub>nd </sub></em>to 0 with probability 1 − <em>p</em>. Essentially, it would be equivalent to replacing each input <em>x</em><em><sub>n </sub></em>by <em>x</em>˜<em><sub>n </sub></em>= <em>x</em><em><sub>n </sub></em>◦<em>m</em><em><sub>n</sub></em>, where ◦ denotes elementwise product and <em>m</em><em><sub>n </sub></em>denotes the <em>D</em>×1 binary mask vector with <em>m<sub>nd </sub></em>∼ Bernoulli(<em>p</em>) (<em>m<sub>nd </sub></em>= 1 means the feature <em>x<sub>nd </sub></em>was retained; <em>m<sub>nd </sub></em>= 0 means the feature <em>x<sub>nd </sub></em>was masked/zeroed).

Let us now define a new loss function using these masked inputs as follows: . Show that minimizing the <em>expected </em>value of this new loss function (where the expectation is used since the mask vectors <em>m</em><em><sub>n </sub></em>are random) is equivalent to minimizing a regularized loss function. Clearly write down the expression of this regularized loss function.

<h1>Problem 3</h1>

(Multi-output Regression with Reduced Number of Parameters) Consider the multi-output regression in which each output <em>y</em><em><sub>n </sub></em>∈ R<em><sup>M </sup></em>in a real-valued vector, rather than a scalar. Assuming a linear model, we can model the outputs as <strong>Y </strong>≈ <strong>XW</strong>, where <strong>X </strong>is the <em>N </em>× <em>D </em>feature matrix and <strong>Y </strong>is <em>N </em>× <em>M </em>response <em>matrix </em>with row <em>n </em>being <em>y</em><em><sub>n</sub></em><sup>&gt; </sup>(note that each column of <strong>Y </strong>denotes one of the <em>M </em>responses), and <strong>W </strong>is the <em>D </em>× <em>M </em>weight matrix, with its <em>M </em>columns containing the <em>M </em>weight vectors <em>w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,…,w</em><em><sub>M</sub></em>. Let’s define a squared error loss function, which is just the usual squared error but summed over all the <em>M </em>outputs.

Firstly, verify that this can also be written in a more compact notation as TRACE[(<strong>Y </strong>− <strong>XW</strong>)<sup>&gt;</sup>(<strong>Y </strong>− <strong>XW</strong>)].

Further, we will assume that the weight matrix <strong>W </strong>can be written as a product of two matrices, i.e., <strong>W </strong>= <strong>BS </strong>where <strong>B </strong>is <em>D </em>× <em>K </em>and <strong>S </strong>is <em>K </em>× <em>M </em>(assume <em>K &lt; </em>min{<em>D,M</em>}). Note that there is a benefit of modeling <strong>W </strong>this way, since now we need to learn only <em>K </em>× (<em>D </em>+ <em>M</em>) parameters as opposed to <em>D </em>× <em>M </em>parameters and, if <em>K </em>is small, this can significantly reduce the number of parameters (in fact, reducing the <em>effective </em>number of parameters to be learned is another way of regularizing a machine learning model). Note (you can verify) that in this formulation, each <em>w</em><em><sub>m </sub></em>can be written as a linear combination of <em>K </em>columns of <strong>B</strong>.

With the proposed representation of <strong>W</strong>, the new objective will be TRACE[(<strong>Y </strong>− <strong>XBS</strong>)<sup>&gt;</sup>(<strong>Y </strong>− <strong>XBS</strong>)] and you need to learn both <strong>B </strong>and <strong>S </strong>by solving the following problem:

{<strong>B</strong><sup>ˆ</sup><em>,</em><strong>S</strong><sup>ˆ</sup>} = argminTRACE[(<strong>Y </strong>− <strong>XBS</strong>)<sup>&gt;</sup>(<strong>Y </strong>− <strong>XBS</strong>)]

<strong>B</strong><em>,</em><strong>S</strong>

We will ignore regularization on <strong>B </strong>and <strong>S </strong>for brevity/simplicity.

Derive an alternating optimization (ALT-OPT) algorithm to learn <strong>B </strong>and <strong>S</strong>, clearly writing down the expressions for the updates of <strong>B </strong>and <strong>S</strong>. Are both subproblems (solving for <strong>B </strong>and solving for <strong>S</strong>) equally easy/difficult in this ALT-OPT algorithm? If yes, why? If no, why not?

Note: Since <strong>B </strong>and <strong>S </strong>are matrices, if you want, please feel free to use results for matrix derivatives (results you will need can be found in Sec. 2.5 of the Matrix Cookbook). However, the problem can be solved even without using matrix derivative results with some rearragement of terms and using vector derivatives.

<h1>Problem 4</h1>

Ridge Regression using Newton’s Method Consider the ridge regression problem:

1           <sup>&gt; </sup>− <strong>X</strong><em>w</em>) +<em>w w</em>ˆ = argmin = argmin             (<em>y </em>− <strong>X</strong><em>w</em>) (<em>y </em><em>ww </em>2

where <strong>X </strong>is the <em>N </em>× <em>D </em>feature matrix and <em>y </em>is the <em>N </em>× 1 vector of labels of the <em>N </em>training examples. Note that the factor of  has been used in the above expression just for convenience of derivations required for this problem and does not change the solution to the problem.

Derive the Newton’s method’s update equations for each iteration. For this model, how many iterations would the Newton’s method will take to converge?

<h1>Problem 5</h1>

(Dice Roll) You have a six-faced dice which you roll <em>N </em>times and record the number of times each of its six faces are observed. Suppose these numbers are <em>N</em><sub>1</sub><em>,N</em><sub>2</sub><em>,…,N</em><sub>6</sub>, respectively. Assume that the probability of a random roll of the dice showning the <em>k<sup>th </sup></em>face (<em>k </em>= 1<em>,</em>2<em>,…,</em>6) to be equal to <em>π<sub>k </sub></em>∈ (0<em>,</em>1).

Assuming an appropriate conjugate prior for the probability vector <em>π </em>= [<em>π</em><sub>1</sub><em>,π</em><sub>2</sub><em>,…,π</em><sub>6</sub>], derive its MAP estimate. In which situation(s), you would expect the MAP solution to be better than the MLE solution?

Also derive the full posterior distribution over <em>π </em>using the same prior that you used for MAP estimate. Given this posterior, can you get the MLE and MAP estimate without solving the MLE and MAP optimization problems? If yes, how? If no, why not?