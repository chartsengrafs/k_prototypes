K-Prototypes: A Case Study
================

\[\alpha, \beta,  \gamma, \Gamma\]

### Erik’s Practical Applications of Unsupervised Learning

1.  Initial exploratory: How many groups of people, places, companies,
    things just naturally exist in this data?
2.  Feature engineering - map multiple features to a smaller, discrete
    categorical variable which can then be plugged in as a more simple
    predictive feature
3.  Target engineering - Once I have found these natural categories
    exist, can I then use them as my Y and predict future membership in
    that cluster?
    <div>

</div>

## A K-Means Refresher

K-Means clustering is a classic algorithm used for clustering that
attempts to find the center of \(k\) specified clusters

## Why K-Prototypes?

The problem with K-Means clustering

You can include R code in the document as follows:

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

## Including Plots

You can also embed plots, for example:

![](k_proto_rmd_files/figure-gfm/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
