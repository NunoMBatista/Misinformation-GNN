## Brief description

Online social networks are prime examples of complex systems, where infor-
mation (credible or not) can spread rapidly through user interactions. In
this project, you will explore the emergence of misinformation cascades by
leveraging Graph Neural Networks (GNNs) on real-world data. Various pub-
licly available Twitter or Facebook datasets (e.g., FakeNewsNet, available on
GitHub) include user–user interactions and post-level content that can be
transformed into graph structures. The goal is to analyze how misleading
posts propagate, identify influential nodes, and examine conditions under
which misinformation dominates or fizzles out.

## Goals

From a publicly available social media dataset, represent user interactions
(e.g., follows, retweets, replies) as graph edges and embed relevant user or
content features as node attributes. The goal is to build a GNN to clas-
sify each news item as fake or real, and compare its performance against a
baseline. Explore which features (e.g., user credibility, network centrality)
the GNN relies on most for detection, and examine whether certain network
structures or activity patterns increase the likelihood of fake news prolifer-
ation. Finally, discuss any threshold or emergent effects that manifest as
misinformation spreads at scale.
