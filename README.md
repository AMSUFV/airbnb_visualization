
# Analysing Brooklyn's airbnb trends


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')
```

## Importing and cleaning the data

[![png](images/nyc_airbnb.png)](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)

We'll use an open source dataset from Kaggle that contains New York's Airbnb data for 2019.


```python
airbnb = pd.read_csv('airbnb.csv', usecols = ['neighbourhood','neighbourhood_group', 'reviews_per_month',
                                              'availability_365','price', 'room_type', 'last_review'])

display(airbnb.head(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>room_type</th>
      <th>price</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>Private room</td>
      <td>149</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>Private room</td>
      <td>150</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Manhattan</td>
      <td>Murray Hill</td>
      <td>Entire home/apt</td>
      <td>200</td>
      <td>2019-06-22</td>
      <td>0.59</td>
      <td>129</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Brooklyn</td>
      <td>Bedford-Stuyvesant</td>
      <td>Private room</td>
      <td>60</td>
      <td>2017-10-05</td>
      <td>0.40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Manhattan</td>
      <td>Hell's Kitchen</td>
      <td>Private room</td>
      <td>79</td>
      <td>2019-06-24</td>
      <td>3.47</td>
      <td>220</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Manhattan</td>
      <td>Upper West Side</td>
      <td>Private room</td>
      <td>79</td>
      <td>2017-07-21</td>
      <td>0.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Manhattan</td>
      <td>Chinatown</td>
      <td>Entire home/apt</td>
      <td>150</td>
      <td>2019-06-09</td>
      <td>1.33</td>
      <td>188</td>
    </tr>
  </tbody>
</table>
</div>


<br>Uppon a quick inspection of the dataset we can find several Nan values in different columns. NaNs tend to be a problem while handling datasets; let's see how widespread they are accross ours:


```python
print(airbnb.isnull().sum())
```

    neighbourhood_group        0
    neighbourhood              0
    room_type                  0
    price                      0
    last_review            10052
    reviews_per_month      10052
    availability_365           0
    dtype: int64
    

<br>There are more than ten thousand NaN values in both the *last_review* column and the *reviews per moth* column. Let's find out how much that is compared to the total number of lines:


```python
print('{:.2f} % NaN values'.format(airbnb.isnull().sum().reviews_per_month*100/len(airbnb)))
```

    20.56 % NaN values
    

We have some ways of dealing with the NaN problem, just to name a few:
- Study where they are coming from; see if they are from a particular neighborhood or if they are split amongst all of them.
- Replace the NaN values with the mean value for that column for their neighborhood.
- Replace the NaN values with the global mean value for that column
- Discard the columns with the NaN values.
- Discard the rows with the NaN values.

Most of them would be time consuming, and dealing with NaNs is not the topic of this notebook. In this case, we'll just discard every row that has a NaN value:


```python
airbnb.dropna(inplace=True)
```

<br>Another good practice is to look for duplicates:


```python
print(airbnb.duplicated().sum())
```

    9
    

We'll discard these rows:


```python
airbnb.drop_duplicates(inplace = True)
```

<br>Now we'll lock for outliers in the data; values really out of place or just plainly wrong. This is easy to do through a boxplot:


```python
fig, axs = plt.subplots(1, 3, figsize=(19, 5))
columns = ['price', 'reviews_per_month', 'availability_365']
for column, ax in zip(columns, axs):
    box = ax.boxplot(airbnb[column], patch_artist=True)
    ax.set_title(column.replace('_', ' ').capitalize())
    # styling
    plt.setp(box['boxes'], color='white')
    plt.setp(box['boxes'], edgecolor='black')
    plt.setp(box['medians'], color='red')
plt.show()
```


![png](images/output_16_0.png)


The *availability* column looks fine; the *price* one, even with its high prices, seems reasonable too (there are some really expensive apartments in NY); but if we take a closer look at the *reviews per month* column we'll be able to see how one apartment has almost 60 reviews per month. And unless that apartment is close to a black hole that distorts time making one day there two days in real life, months have 31 days at max.

*Note: Last time I checked, Airbnb reservations are made by one person only and you have to book the place for the night, hence the consideration of that value as an outlier.*

We'll get rid of that outlier:


```python
airbnb = airbnb[airbnb.reviews_per_month <= 31]
```

<br>

### Williambsurg
Williamsburg is one of the trendiest neighborhoods nowadays. Since 2005 it has changed from an industrial district to a place that houses many young people and blooming businesses. We'll analyse how it does on Airbnb compared with the rest of the neighborhoods; for that, we'll take into account the *availability* and the *reviews per month*, which intuitively will let us know how solicited different neighborhoods are and how people rate them.

First, we'll categorise the listings by the ammount of reviews per month (rpm) they get. We'll split them into three categories:
- Low ammount of rpm.
- Medium ammount of rpm.
- High ammount of rpm.


```python
# We create a new column to store the tag 
airbnb['reviews_tag'] = ''

# And divide the data using quantiles
quantiles = airbnb['reviews_per_month'].quantile([0.33, 0.66])

airbnb.loc[(airbnb.reviews_per_month > 0 ) & (airbnb.reviews_per_month <= quantiles.values[0]), 'reviews_tag'] = 'Low'
airbnb.loc[(airbnb.reviews_per_month > quantiles.values[0] ) & (airbnb.reviews_per_month <= quantiles.values[1]), 'reviews_tag'] = 'Medium'
airbnb.loc[(airbnb.reviews_per_month > quantiles.values[1] ), 'reviews_tag'] = 'High'
```

<br>Now we'll classify the listings based on their availability in a similar manner; we'll use the following tags:
- Low demand
- Average demand
- High demand


```python
airbnb['availability_tag'] = ''

quantiles = airbnb['availability_365'].quantile([0.33, 0.66])

airbnb.loc[(airbnb.availability_365 <= quantiles.values[0]), 'availability_tag'] = 'High demand'
airbnb.loc[(airbnb.availability_365 > quantiles.values[0] ) & (airbnb.availability_365 <= quantiles.values[1]), 'availability_tag'] = 'Average demand'
airbnb.loc[(airbnb.availability_365 > quantiles.values[1] ), 'availability_tag'] = 'Low demand'
```

<br>Lastly, we'll relate the number of reviews with the availability to obtain a rating for the listing; if it's on high demand and it gets plenty of reviews, it's safe to assume its a very good place. On the oder hand, if it has low demand and has a high number of reviews per month, the place is probably not the best.

We'll use the following tags:
- Very good
- Good
- Average
- Bad
- Very bad


```python
airbnb['classification'] = ''
classification_tags = ['Very good', 'Good', 'Average', 'Bad', 'Very bad']

airbnb.loc[(airbnb['reviews_tag'] == 'Low') & (airbnb['availability_tag'] == 'High demand'), 'classification'] = 'Good'
airbnb.loc[(airbnb['reviews_tag'] == 'Medium') & (airbnb['availability_tag'] == 'High demand'), 'classification'] = 'Good'
airbnb.loc[(airbnb['reviews_tag'] == 'High') & (airbnb['availability_tag'] == 'High demand'), 'classification'] = 'Very good'

airbnb.loc[(airbnb['reviews_tag'] == 'Low') & (airbnb['availability_tag'] == 'Average demand'), 'classification'] = 'Average'
airbnb.loc[(airbnb['reviews_tag'] == 'Medium') & (airbnb['availability_tag'] == 'Average demand'), 'classification'] = 'Average'
airbnb.loc[(airbnb['reviews_tag'] == 'High') & (airbnb['availability_tag'] == 'Average demand'), 'classification'] = 'Good'

airbnb.loc[(airbnb['reviews_tag'] == 'Low') & (airbnb['availability_tag'] == 'Low demand'), 'classification'] = 'Bad'
airbnb.loc[(airbnb['reviews_tag'] == 'Medium') & (airbnb['availability_tag'] == 'Low demand'), 'classification'] = 'Bad'
airbnb.loc[(airbnb['reviews_tag'] == 'High') & (airbnb['availability_tag'] == 'Low demand'), 'classification'] = 'Very bad'
```

<br>Let's see the changes:


```python
display(airbnb.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>room_type</th>
      <th>price</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>availability_365</th>
      <th>reviews_tag</th>
      <th>availability_tag</th>
      <th>classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>Private room</td>
      <td>149</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>365</td>
      <td>Low</td>
      <td>Low demand</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>355</td>
      <td>Medium</td>
      <td>Low demand</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>194</td>
      <td>High</td>
      <td>Low demand</td>
      <td>Very bad</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>0</td>
      <td>Low</td>
      <td>High demand</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Manhattan</td>
      <td>Murray Hill</td>
      <td>Entire home/apt</td>
      <td>200</td>
      <td>2019-06-22</td>
      <td>0.59</td>
      <td>129</td>
      <td>Medium</td>
      <td>Average demand</td>
      <td>Average</td>
    </tr>
  </tbody>
</table>
</div>


<br>Let's take a look at the top 10 neighborhoods in Brooklyn so we can see how Williamsburg compares:


```python
brooklyn = airbnb.loc[airbnb.neighbourhood_group == 'Brooklyn', :]

top_10 = brooklyn['neighbourhood'].value_counts()[:10]

# Neighborhood names
top_10_names = top_10.index

# Their review count
top_10_nreviews = top_10.values

# This piece of code gets the number of reviews and puts them in a list for a later use in a graph
scores = []
for neighborhood in top_10_names:
    # For each neighborhood in the top 10, we get the ammount of classifications of each type
    reviews = airbnb.loc[airbnb.neighbourhood == neighborhood, 'classification'].value_counts().reindex(classification_tags).values
    scores.append(reviews)
scores = np.array(scores)

# Normalized scores 
scores_norm = scores * 100 / scores.sum(axis=1)[:, None]
```


```python
fig, ax = plt.subplots(2,1, figsize = (14,14))

# Number of reviews
ax[0].barh(top_10_names, top_10_nreviews)
ax[0].set_xlabel('Number of reviews', fontsize = 13)
ax[0].set_title('Top 10 neighborhoods in Brooklyn by review count', fontsize = 16)

# Type of reviews
scores_norm_cum = scores_norm.cumsum(axis=1)
colors = ['#B3CC57','#ECF081', '#FFBE40', '#EF746F', '#AB3E5B']
for i, (reviews, color) in enumerate(zip(scores, colors)):
    if i<5:
        widths = scores_norm[:,i]
        starts = scores_norm_cum[:,i] - widths
        ax[1].barh(top_10_names, widths, left = starts, height = 0.6, color = color)

ax[1].legend(classification_tags,
             loc = 'best', bbox_to_anchor = (1,1))
ax[1].set_xlabel('Type of review', fontsize = 13)
ax[1].set_xticks(np.arange(0,110,10))
plt.show()
```


![png](images/output_29_0.png)


These two plots show the busiest neighborhoods in Brooklyn based on the number of reviews. You can appreciate how Williamsburg is the neighborhood with the highnest number of reviews, closely follow by Bedford-Stuyvesant.

Even though these two neighborhoods have a similar ammount of reviews, if we take a look at the second plot we are able to see that Bedford has close to 20% more negative reviews than Williamsburg; the former has close to 40% negative reviews while the latter close to 20%.

There are two interesting neighborhoods based on the second plot: Williamsburg, for having the highest number of reviews and the highest proportion of possitive reviews, and East Flatbush, where the number of negative reviews is higher than that of the possitive reviews and the neutral reviews added.

<br>

### Taking a look at the Top 5

We'll now group the different neighborhoods by *classification*, keeping the 5 with the highest proportion of *Very good* reviews:


```python
airbnb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>room_type</th>
      <th>price</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>availability_365</th>
      <th>reviews_tag</th>
      <th>availability_tag</th>
      <th>classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>Private room</td>
      <td>149</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>365</td>
      <td>Low</td>
      <td>Low demand</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>355</td>
      <td>Medium</td>
      <td>Low demand</td>
      <td>Bad</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>194</td>
      <td>High</td>
      <td>Low demand</td>
      <td>Very bad</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>0</td>
      <td>Low</td>
      <td>High demand</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Manhattan</td>
      <td>Murray Hill</td>
      <td>Entire home/apt</td>
      <td>200</td>
      <td>2019-06-22</td>
      <td>0.59</td>
      <td>129</td>
      <td>Medium</td>
      <td>Average demand</td>
      <td>Average</td>
    </tr>
  </tbody>
</table>
</div>




```python
neighborhood_data = airbnb.groupby('neighbourhood')['classification'].value_counts().unstack().fillna(0)

top_5 = neighborhood_data.sort_values(by='Very good', ascending = False)[:5]
top_5 = top_5[classification_tags]
```

<br>Let's now add the relevant information to each neighborhood (average price, monthly reviews and availability):


```python
top_5_data = airbnb.groupby(['neighbourhood', 'classification']).mean()
top_5_data = top_5_data.loc[list(top_5.index),:]
display(top_5_data)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>price</th>
      <th>reviews_per_month</th>
      <th>availability_365</th>
    </tr>
    <tr>
      <th>neighbourhood</th>
      <th>classification</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Bedford-Stuyvesant</th>
      <th>Average</th>
      <td>109.784615</td>
      <td>0.643934</td>
      <td>54.663736</td>
    </tr>
    <tr>
      <th>Bad</th>
      <td>118.788889</td>
      <td>0.610815</td>
      <td>286.438889</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>97.705202</td>
      <td>1.440918</td>
      <td>25.439306</td>
    </tr>
    <tr>
      <th>Very bad</th>
      <td>114.758140</td>
      <td>3.001256</td>
      <td>270.220155</td>
    </tr>
    <tr>
      <th>Very good</th>
      <td>80.991453</td>
      <td>2.749145</td>
      <td>0.102564</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">East Harlem</th>
      <th>Average</th>
      <td>119.462585</td>
      <td>0.701633</td>
      <td>55.795918</td>
    </tr>
    <tr>
      <th>Bad</th>
      <td>138.109489</td>
      <td>0.634088</td>
      <td>273.131387</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>135.644540</td>
      <td>1.706510</td>
      <td>27.569593</td>
    </tr>
    <tr>
      <th>Very bad</th>
      <td>143.063830</td>
      <td>2.990071</td>
      <td>258.163121</td>
    </tr>
    <tr>
      <th>Very good</th>
      <td>102.288462</td>
      <td>3.279038</td>
      <td>0.134615</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Harlem</th>
      <th>Average</th>
      <td>140.229167</td>
      <td>0.636589</td>
      <td>60.968750</td>
    </tr>
    <tr>
      <th>Bad</th>
      <td>134.755396</td>
      <td>0.543285</td>
      <td>282.633094</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>99.512974</td>
      <td>1.328054</td>
      <td>21.726547</td>
    </tr>
    <tr>
      <th>Very bad</th>
      <td>127.177019</td>
      <td>2.934534</td>
      <td>255.701863</td>
    </tr>
    <tr>
      <th>Very good</th>
      <td>93.888889</td>
      <td>2.587654</td>
      <td>0.148148</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Upper West Side</th>
      <th>Average</th>
      <td>202.295276</td>
      <td>0.559764</td>
      <td>55.169291</td>
    </tr>
    <tr>
      <th>Bad</th>
      <td>256.527157</td>
      <td>0.427572</td>
      <td>281.929712</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>155.208895</td>
      <td>0.933518</td>
      <td>13.057951</td>
    </tr>
    <tr>
      <th>Very bad</th>
      <td>193.975410</td>
      <td>3.148197</td>
      <td>261.901639</td>
    </tr>
    <tr>
      <th>Very good</th>
      <td>118.705882</td>
      <td>2.637451</td>
      <td>0.137255</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Williamsburg</th>
      <th>Average</th>
      <td>152.343907</td>
      <td>0.623573</td>
      <td>49.784641</td>
    </tr>
    <tr>
      <th>Bad</th>
      <td>163.973558</td>
      <td>0.540601</td>
      <td>275.733173</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>123.576531</td>
      <td>1.002177</td>
      <td>16.390590</td>
    </tr>
    <tr>
      <th>Very bad</th>
      <td>185.812721</td>
      <td>3.090883</td>
      <td>270.038869</td>
    </tr>
    <tr>
      <th>Very good</th>
      <td>122.278351</td>
      <td>2.419485</td>
      <td>0.082474</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_reviews = list(top_5.columns)[::-1]

colors = ['#B3CC57','#ECF081', '#FFBE40', '#EF746F', '#AB3E5B']
plots = ['price', 'reviews_per_month', 'availability_365']
colors = colors[::-1]

fig, ax = plt.subplots(1,3, figsize=(20,5))
for i, plot in enumerate(plots):

    
    sizes = []
    for review in plot_reviews:
        sizes.append(top_5_data.xs(review, level='classification', drop_level = False)[plot].values)
    
    # up-scaling for a better view
    sizes = np.array(sizes)
    sizes = (200 * sizes) / sizes.max()
        
    for j, (size, color) in enumerate(zip(sizes, colors)):
        x = np.arange(0, len(size), 1)
        y = np.full(len(sizes), j)
        ax[i].scatter(x, y, size, c = color)

    ax[i].set_yticks(np.arange(5))
    ax[i].set_xticks(np.arange(5))
    ax[i].set_xticklabels(top_5.index, rotation = 'vertical')
    ax[i].set_yticklabels(plot_reviews)

    ax[i].set_title(plot.replace('_', ' ').capitalize(), fontsize = 13)
    
plt.show()
```


![png](images/output_37_0.png)


From left to right, these plots show the correlation between price, monthly reviews and availiability with the neighborhood and the rating for the neighborhood, where the size of the bubble meaning higher or lower price, number of reviews or availability, and the color being associated with the type of review (green for very good, yellow for good, etc.).

Starting with the plot on the left, we can say that the price difference between the highest and the lowest rated places is not that high, although we can see how higher rated places tend to be cheaper than lower rated ones. It's pretty likely that places reviewed as *Very bad* were expensive and failed to meet the associated expectations in quality, hence the review. On the other hand, those with the best score have lower places, which can indicate that they probably satisfied the expectations for a lower priced place.

The plot in the middle shows some horizontal symmetry; it looks like the users are more likely to voice their opinions when the place has left a strong impression in them, either a really good or a really bad one. Users tend to speak up to recommend places they have enjoyed and discourage others from staying at places where they've had a bad time. On the other hand, places reviewed from *Good* to *Average* show a smaller number of reviews, which supports the strong impression hypothesis.

Finally, the plot in the right shows a really strong, easy to interpret gradient; the quality of the place is closely related to its availability. We can (barely) see how the best rated places are rarely available, while the opposite is true for the worst rated ones.

<br>

### What to rent? Comparing prices

Now that we now where to rent (and where not to), let's try to understand the differences between renting an apartment, a shared room and a private room:


```python
top_5_data = airbnb.groupby(['neighbourhood', 'room_type']).mean()
top_5_data = top_5_data.loc[list(top_5.index),:]
top_5_data.reset_index(inplace=True)
```


```python
fig, axs = plt.subplots(1,3, figsize=(20,5))

columns = ['price', 'reviews_per_month', 'availability_365']
for ax, column in zip(axs, columns):
    sns_plot = sns.barplot(x='neighbourhood', y=column, hue='room_type', data=top_5_data, ax=ax)
    # styling
    sns_plot.set_title(column.replace('_', ' ').capitalize())
    sns_plot.set_ylabel(column.replace('_', ' ').capitalize())
    sns_plot.set_xlabel('Neighborhood')
    sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=45)
    sns_plot.legend(title='Room type:')
    
plt.show()
```


![png](images/output_42_0.png)


In the first plot we are able to see how it's way more expensive (almost double the price in most cases) to rent an entire home/apt than renting a private or a share room, with the last one being the cheapest option.

The fact that renting an entire home/apt is the most expensive option doesn't prevent it from being the prefered one; if we take a look at the third plot, renting the whole apt is the prefered option together with renting a private room; it's safe to say that users value their privacy and will pay additional money to keep it.
