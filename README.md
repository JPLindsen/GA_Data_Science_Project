# The Film Poster Project

Ever noticed that the design of film posters is often very stereotypical? Although many good film posters are designed beautifully, they always have a clear commercial purpose - to promote a film. To do so, the target audience must be able to understand quickly what the movie is about. In other words, the design of the poster must convey some important aspects of the movie, such as the genre.

In this project I looked at the relationship between low-level properties, such as colour and amount spatial detail, and genre. I extracted these low-level features from a large set of film posters scraped from IMDB, and subsequently used those to predict the film’s genre. If these low-level features differ systematically between film genres, my prediction was that the accuracy of the prediction should be above chance level.

![Illustration of some features extracted using the Toy Story 3 poster.](https://drive.google.com/uc?export=view&id=0B1M_z8zxOEmbck1HQzRoMVgtLTQ)

With 7 genres (Horror, Romance, Comedy, Sci-Fi, Thriller, Action, and Family), the best performing model was able to predict genre correctly 31% of the time, just under one-in-three. Although this score was considerably higher than chance level, it still is some way off from making consistently making reliable decisions. Interestingly, a deeper investigation of the prediction results showed that misclassification between genres was mostly within the subsets [horror, action, sci-fi, thriller] and [family, comedy, romance]. This could be associated with the darker/brighter atmosphere associated with those two sets.

![Boxplots of the Results of the 10-fold Cross-Validation, separately for the six models tested. The vertical red line indicates chance level](https://drive.google.com/uc?export=view&id=0B1M_z8zxOEmbaDZ6M2VqV0hFZVU)

Conclusion? The genre of a film can be to some extend predicted from low-level properties of its film poster, but classification seems to favour a more simple 2-class separation presumably based on brightness of the poster.

=============================
Python Libaries used:
- sklearn
- numpy
- pandas
- skimage
- matplotlib
- BeautifulSoup
- urllib

=============================
This work was one as part of my final project for the excellent [Data Science](https://generalassemb.ly/education/data-science) course at [General Assembly](https://generalassemb.ly/).
