# Logistic Regression Optimization Project

## Project Description

This lab investigates logistic regression for a **multi-class classification task** using both `scikit-learn` and a custom-built logistic regression classifier. The custom implementation allows the use of different optimizers and regularization techniques, and follows object-oriented principles similar to scikit-learn's API.

---

## Business Use Case

We use player statistics from the [FIFA 22 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) to predict a player's wage class. This model helps **football club managers** determine if they're overpaying or underpaying players based on similar players' statistics and market value.

- **Intended Users**: Top-level club management  
- **Deployment Strategy**: Private, offline model used internally  
- **Required Accuracy**: 85%+ for practical application  
- **Model Type**: Multi-class classification (4 classes: low, medium, high, super)

---

## Dataset Description

- **Source**: FIFA 22 dataset from Kaggle
- **Preprocessing**:
  - Nulls replaced with `0`
  - Added categorical `wage_class` label (0 to 3)
- **Size**: 61 features, 19,239 rows
- **Target Variable**: `wage_class`
- **Class Distribution**: Majority in "high" and "super" classes
- **Observation**: Wage correlates with player age and longevity

---

## Model Design

### 1. **Baseline - scikit-learn Logistic Regression**
- Accuracy (80/20 split): `~0.92`
- Cross-validation Accuracy: Mean = `0.92`, Std = `0.00`
- Runtime: ~836 ms

### 2. **Custom Logistic Regression**
- Built using `numpy` and `scipy`
- Implements:
  - **Steepest Ascent**
  - **Stochastic Gradient Ascent**
  - **Newton’s Method**
- Includes support for:
  - **No regularization**
  - **L1**, **L2**, and **ElasticNet**
  - Configurable regularization strength `C`

#### Best Performing Configuration:
- Optimizer: `normal`
- Regularization: `l1`
- C: `0.001`
- Accuracy: `0.9108`
- Runtime: ~5.1 seconds

---

## Parameter Tuning

We performed grid search on:
- Optimizers: `normal`, `steepest`, `stochastic`, `newton_method`
- Regularization: `none`, `l1`, `l2`, `elasticnet`
- Regularization strength `C`: `0.001` to `100`

### Key Insights:
- **Newton's Method** underperformed (accuracy < 0.45)
- **Stochastic Gradient Ascent** showed inconsistent results
- **Steepest Ascent** and `normal` gave the best and most stable accuracy
- Visualizations include heatmaps, boxplots, and 3D plots

---

## Performance Comparison

| Method              | Accuracy | Runtime       |
|---------------------|----------|---------------|
| Scikit-learn        | 0.9226   | ~836 ms       |
| Custom (best config)| 0.9108   | ~5.1 seconds  |

### Verdict:
Despite good performance, our custom model is **slower** and only marginally less accurate. For any production deployment, scikit-learn should be preferred.

---

## Deployment Recommendation

We **strongly recommend** using `scikit-learn` for deployment due to:

- **Speed and Optimization**: Written in C++ and optimized for production
- **Reliability**: Fewer bugs, broad community support
- **Documentation**: Robust and well-maintained
- **Accuracy**: Comparable or better than our implementation
- **Maintainability**: Easier to update and validate over time

Our custom code is a good learning exercise, but not suitable for real-world deployment.
