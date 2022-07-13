![project banner](https://project-banner.phamn23.repl.co/?title=Chicken+Research&description=A+bad+attempt+at+determining+the+weight+of+a+chicken&stack=python)

# Chicken Research

A bad attempt at determining the weight of a chicken based on a ratio of the area of a "magic" blue square to the area of the chicken obtained through image segmentation. See the [research paper](https://docs.google.com/document/d/1aqCSacu-UWtq8FQwx99H199qJym9Z8Lk5wKOiC_j0DE/edit?usp=sharing) on Google Docs.

## Methodology

1. take image of chicken with blue square on its back
2. put image into `chicken_train`
3. run augmentation on images & resize to width = 300
4. train model

Note: full dataset not added
