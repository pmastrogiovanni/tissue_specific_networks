# FunCoup-like inference of functional association networks from tissue specific data

The FunCoup network is based on a global model of protein functional associations which integrates hundreds of experimental datasets by a data-driven Bayesian framework, without distinction between tissues of origin. Tissue-specific networks can find disease-related relationships and disclose the varyingfunctional roles of genes across tissues. 

Some attempts to model tissue-specific functional association networks were done by applying filters on genome-scale at pre- or post-training phase: (i) GIANT infers tissue-specific networks from genome-wide generic data using filtered gold standard interactions via tissue-specific annotations; (ii) FunCoup5 implements a filter to visualize pre-computed networks with only genes expressed in specific tissues according to protein abundance in the Human Protein Atlas. 

In this project, we attempted to infer tissue-specific networks starting from single cell RNA sequencing data, which holds high cell resolution and allows to extract tissue-specific functional associations. Inference was performed on seven human tissues (ovary, heart, pancreas, adrenal gland, muscle, lung, liver) applying the FunCoup’s Bayesian framework, which establishes link confidence based on four gold standards.

Resulting networks were analysed mainly to assess coverage, statistical significance, and biological specificity. Coverage was addressed by evaluating network features, such as dimension and node connectivity, after filtering for high confidence links. By computing similarity measures between networks for nodes, edges and significantly connected pathways, we highlighted differences in their overall gene interactions, showing the feasibility of this approach. 

Whileacknowledging the current limitations, such as the lack of a state-of-the-art approach for scRNA-seq quality control and the overall poor data availability, this research provides a foundation for future investigations.
