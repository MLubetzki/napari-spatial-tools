# Some more features to implement
- Differential score: You feed in two lists of genes (diffex selection of two cell types), choose a colormap that has two good extremes and then calculate the scores and plot their difference or sth. For telling apart two closely related cell types
- Multi-color plotting (for different markers)
- Make the API easier. You don't need separate functions with their own parameters. You can probably merge most into a single plot function. Move score calculation to a separate function?
- Assume we plotted a score and found cells that rank high in the score - can we give the user an easy way to see the genes that contributed the most to the score in this cell?
- Implement total count normalization or do it before and implement layers selection. Scores and expressions are currently heavily biased by total counts in the spot
- Would be nice to mark an area and run diffex vs the rest
- IIRC there was a UMAP function in the plugin?