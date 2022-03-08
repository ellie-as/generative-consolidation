
### A generative computational model of consolidation

Code for modelling systems consolidation as teacher-student learning, in which initial representations of memories are replayed to train a generative model.

To use this code, clone the repo and launch jupyter lab. Each notebook installs its requirements in the first cell (however in the case of installation issues, I recommend using AWS SageMaker).

#### Subfolders

* **main_results**: code for modelling consolidation as teacher-student learning, with a modern Hopfield network teacher (representating the initial hippocampal encoding), and a VAE student (representing the generative model trained by consolidation). Also includes code for testing novelty-mediated consolidation, and code for testing how both the teacher and student network could contribute to recall of a partially consolidated memory.
* **additional_distortions**: other memory distortion results (e.g. modelling the Bartlett (1932) and Deese-Roediger-McDermott results)
* **shapes_vae**: code for exploring the latent space of a VAE trained on the shapes3d dataset, and testing how latent representations may support semantic memory.
* **sequence_learning**: code for exploring how the results may extend to memory for sequences (including text and paths on a graph), and how consolidation may improve structural inference.
* **colab_versions**: Colab-compatible versions of the notebooks, corresponding to the Colab links below. Ignore if not using Colab.

#### Colab notebooks

To enable experimentation with the code without needing to install everything locally, Colab versions are provided for a few of the notebooks (this requires a Google Drive account). Click the link to go straight to the notebook:
* End-to-end simulation, which reproduces the main results for the MNIST dataset: https://colab.research.google.com/github/eas-93/generative-consolidation/blob/master/colab_versions/Consolidation_simulation.ipynb
* Explore the latent space of a trained VAE: https://colab.research.google.com/github/eas-93/generative-consolidation/blob/master/colab_versions/shapes_vae-latent_space.ipynb
* Explore how latent variable representations could support semantic memory: https://colab.research.google.com/github/eas-93/generative-consolidation/blob/master/colab_versions/shapes_vae-semantic_knowledge.ipynb
* Train an autoregressive model on random walks in a graph, and show how this produces structural inference: https://colab.research.google.com/github/eas-93/generative-consolidation/blob/master/colab_versions/Graph_sequence_model.ipynb

Note that these may be slow if running on CPU. To switch to GPU in Colab, go to 'Runtime' > 'Change runtime type', and select 'GPU' from the 'Hardware acceleration' dropdown menu. (Note that the free version of Colab will only allow this for one notebook at a time.)

#### Figures in paper

Figure | Corresponding code
--- | ---
Figure 3 | [./main_results/Consolidation_simulation.ipynb](./main_results/Consolidation_simulation.ipynb)
Figure 4 | [./main_results/Consolidation_simulation.ipynb](./main_results/Consolidation_simulation.ipynb)
Figure 5 | [./main_results/Consolidation_simulation.ipynb](./main_results/Consolidation_simulation.ipynb)
Figure 6 | [./main_results/Memory_distortions.ipynb](./main_results/Memory_distortions.ipynb)
Figure 8 | [./additional_distortions/DRM_VAE_and_AE.ipynb](./additional_distortions/DRM_VAE_and_AE.ipynb)
Figure 9 | [./shapes_vae/shapes_vae-latent_space.ipynb](./shapes_vae/shapes_vae-latent_space.ipynb)
Figure 10 | [./shapes_vae/shapes_vae-latent_space.ipynb](./shapes_vae/shapes_vae-latent_space.ipynb)
Figure 11 | [./shapes_vae/shapes_vae-latent_space.ipynb](./shapes_vae/shapes_vae-latent_space.ipynb)
Figure 12 | [./shapes_vae/shapes_vae-semantic_knowledge.ipynb](./shapes_vae/shapes_vae-semantic_knowledge.ipynb)
Figure 13 | [./main_results/Novelty_and_consolidation.ipynb](./main_results/Novelty_and_consolidation.ipynb)
Figure 14 | [./main_results/Hybrid_recall.ipynb](./main_results/Hybrid_recall.ipynb)
Figure 15 | [./sequence_learning/Durrant_sequence_model.ipynb](./sequence_learning/Durrant_sequence_model.ipynb)
Figure 16 | [./sequence_learning/Graph_sequence_model.ipynb](./sequence_learning/Graph_sequence_model.ipynb)


