---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
Alberto Rodriguez


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.

```python
import numpy as np
import math
import pandas as pd
```

## Exercise 1

```python
a = np.full(shape=(6,4), fill_value=2)
a
```

## Exercise 2

```python
b = np.full(shape=(6,4), fill_value=1)
np.fill_diagonal(b, 3)
# b[range(4), range(4)] = 3
b
```

## Exercise 3

```python
a * b
```

```python
# np.dot(a, b) breaks
```

Literally multiplying each (row,col) value within their respective matrices does work (because they are the same shape). However, this does not work for the dot product because the b's rows does not match a's columns.


## Exercise 4

```python
print("Shape of Matrices: ", a.transpose().shape, "x", b.shape)
answer = np.dot(a.transpose(), b)
print("Resulting Shape: ", answer.shape)
answer
```

```python
print("Shape of Matrices: ", a.shape, "x", b.transpose().shape)
answer = np.dot(a, b.transpose())
print("Resulting Shape: ", answer.shape)
answer
```

The calculations above are valid because a's columns are adjusted to match b's rows, and vica versa. Although changing a or b both allow for the dot product to work, they result in different shapes since the number of a's rows is preseved as well as b's columns.


## Exercise 5

```python
def totallyNotAFunction():
    print("Just kidding this is a function")
totallyNotAFunction()
```

## Exercise 6

```python
def randomArrayStats():
    arr_len = np.random.randint(low=1, high=100, size=1)[0]
    arr = np.random.rand(arr_len, 1)
    print("Sum: ", arr.sum())
    print("Mean: ", arr.mean())
    print("Median: ", np.median(arr))
randomArrayStats()
```

## Exercise 7

```python
def count_ones_loop(arr):
    count = 0
    for value in arr:
        if value == 1:
            count = count + 1
    return count
assert count_ones_loop([1, 2, 3, 1]) == 2

def count_ones_where(arr_like):
    arr = np.array(arr_like)
    return np.where(arr==1, arr, 0).sum()

assert count_ones_where([1, 2, 3, 1]) == 2, "Oh no"
assert count_ones_where([2, 3, 4]) == 0, "Oh no x2"
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
a_np = np.full(shape=(6,4), fill_value=2)
a = pd.DataFrame(a_np)
a
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
b_np = np.full(shape=(6,4), fill_value=1)
np.fill_diagonal(b_np, 3)
b = pd.DataFrame(b_np)
# b.iloc[range(4), range(4)] = 3
b
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
a * b
```

```python
# np.dot(a, b)
```

The line above does not work for the same reasons as before, the shapes do not satisfy the requirements for the dot product operation.


## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
def count_ones_loop(df):
    count = 0
    for row_index, columns in df.iterrows():
        for col_index in columns:
            if df.iloc[col_index][row_index] == 1:
                count = count + 1
    return count

assert count_ones_loop(b) == 20
assert count_ones_loop(a) == 0
    
def count_ones_where(df):
    return df.where(df == 1, 0).sum().sum()

assert count_ones_where(b) == 20
assert count_ones_where(a) == 0, count_ones_where(a)
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df.head(1)
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df["name"]
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df.set_index('sex',inplace=True)
```

titanic_df.loc["female"] is how one would use the new index to locate the female passengers.

```python
print("Number of female passengers: ", len(titanic_df.loc["female"]))
```

## Exercise 14
How do you reset the index?

```python
titanic_df.reset_index(inplace=True)
```

```python
titanic_df.head()
```

```python

```
