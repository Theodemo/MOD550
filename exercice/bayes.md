# Introduction to Bayesian thinking and modeling
## Bayes's Rule

We have already derived Bayes's Rule:

$$P(A|B) = \frac{P(A) P(B|A)}{P(B)}$$

As an example, we used an oil drilling example and Bayes's Rule  to compute conditional probability of finding oil if the seismic indicated that the prospect holds oil. We could also compute the probability of finding oil if the seismic indicated that the prospect **does not** hold oil as well as the probabilities of the prospect **not holding oil** if the seismic indicated "oil" or "dry" (not oil). 

In this chapter, we'll use it to solve several more challenging problems using conditional probability in the form of Bayes Rule.
## The Cookie Problem


We'll start with a simple case:

> Suppose there are two bowls of cookies.
>
> * Bowl 1 contains 30 vanilla cookies and 10 chocolate cookies. 
>
> * Bowl 2 contains 20 vanilla cookies and 20 chocolate cookies.
>
> Now suppose you choose one of the bowls at random and, without looking, choose a cookie at random. If the cookie is vanilla, what is the probability that it came from Bowl 1?

What we want is the conditional probability that we chose from Bowl 1 given that we got a vanilla cookie, $P(B_1 | V)$.

But what we get from the statement of the problem is:

* The conditional probability of getting a vanilla cookie, given that we chose from Bowl 1, $P(V | B_1)$ and

* The conditional probability of getting a vanilla cookie, given that we chose from Bowl 2, $P(V | B_2)$.

Bayes's Rule tells us how they are related:

$$P(B_1|V) = \frac{P(B_1)~P(V|B_1)}{P(V)}$$

The term on the left is what we want. The terms on the right are:

-   $P(B_1)$, the probability that we chose Bowl 1,
    unconditioned by what kind of cookie we got. 
    Since the problem says we chose a bowl at random, 
    we set $P(B_1) = 1/2$.

-   $P(V|B_1)$, the probability of getting a vanilla cookie
    from Bowl 1, which is 3/4.

-   $P(V)$, the probability of drawing a vanilla cookie from
    either bowl. 

To compute $P(V)$, we can use the law of total probability:

$$P(V) = P(B_1)~P(V|B_1) ~+~ P(B_2)~P(V|B_2)$$

Plugging in the numbers from the statement of the problem, we have

$$P(V) = (1/2)~(3/4) ~+~ (1/2)~(1/2) = 5/8$$

We can also compute this result directly, like this: 

* Since we had an equal chance of choosing either bowl and the bowls contain the same number of cookies, we had the same chance of choosing any cookie. 

* Between the two bowls there are 50 vanilla and 30 chocolate cookies, so $P(V) = 5/8$.

Finally, we can apply Bayes's Rule to compute the posterior probability of Bowl 1:

$$P(B_1|V) = (1/2)~(3/4)~/~(5/8) = 3/5$$

This example demonstrates one use of Bayes's theorem: it provides a
way to get from $P(B|A)$ to $P(A|B)$. 
This strategy is useful in cases like this where it is easier to compute the terms on the right side than the term on the left.

## Diachronic Bayes

There is another way to think of Bayes's theorem: it gives us a way to
update the probability of a hypothesis, $H$, given some body of data, $D$.

This interpretation is "diachronic", which means "related to change over time"; in this case, the probability of the hypotheses changes as we see new data.

Rewriting Bayes's rule with $H$ and $D$ yields:

$$P(H|D) = \frac{P(H)~P(D|H)}{P(D)}$$

In this interpretation, each term has a name:

-  $P(H)$ is the probability of the hypothesis before we see the data, called the prior probability, or just **prior**.

-  $P(H|D)$ is the probability of the hypothesis after we see the data, called the **posterior**.

-  $P(D|H)$ is the probability of the data under the hypothesis, called the **likelihood**.

-  $P(D)$ is the **total probability of the data**, under any hypothesis.

Sometimes we can compute the prior based on background information. For example, the cookie problem specifies that we choose a bowl at random with equal probability.

In other cases the prior is subjective; that is, reasonable people might disagree, either because they use different background information or because they interpret the same information differently.

The likelihood is usually the easiest part to compute. In the cookie
problem, we are given the number of cookies in each bowl, so we can compute the probability of the data under each hypothesis.

Computing the total probability of the data can be tricky. 
It is the probability of seeing the data under any hypothesis at all. 
Most often we work through this by specifying a set of hypotheses that
are:

* Mutually exclusive, which means that only one of them can be true, and

* Collectively exhaustive, which means one of them must be true.

When these conditions apply, we can compute $P(D)$ using the law of total probability.  For example, with two hypotheses, $H_1$ and $H_2$:

$$P(D) = P(H_1)~P(D|H_1) + P(H_2)~P(D|H_2)$$

And more generally, with any number of hypotheses:

$$P(D) = \sum_i P(H_i)~P(D|H_i)$$

The process in this section, using data and a prior probability to compute a posterior probability, is called a **Bayesian update**.

## Bayes Tables

A convenient tool for doing a Bayesian update is a Bayes table.
You can write a Bayes table on paper or use a spreadsheet, but in this section I'll use a Pandas `DataFrame`.

First I'll make empty `DataFrame` with one row for each hypothesis:


```python
import pandas as pd
# Define the hypotheses (choosing from Bowl 1 or Bowl 2)
hypotheses = ["Bowl 1", "Bowl 2"]
# Create an empty DataFrame with one row per hypothesis
bayes_table = pd.DataFrame(index=hypotheses)
# Add prior probabilities (P(H))
bayes_table["Prior"] = [1/2, 1/2]
# Add likelihoods (P(D|H)) - probability of drawing a vanilla cookie
bayes_table["Likelihood"] = [3/4, 1/2]

```

Here we see a difference from the previous method: we compute likelihoods for both hypotheses, not just Bowl 1:

* The chance of getting a vanilla cookie from Bowl 1 is 3/4.

* The chance of getting a vanilla cookie from Bowl 2 is 1/2.

As you see, the likelihoods don't add up to 1.  That's OK; each of them is a probability conditioned on a different hypothesis.
There's no reason they should add up to 1 and no problem if they don't.

The next step is similar to what we did with Bayes's Rule; we multiply the priors by the likelihoods: