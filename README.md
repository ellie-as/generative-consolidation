
### A generative computational model of consolidation

Code for modelling consolidation as teacher-student learning, in which initial representations of memories are replayed to train a generative model.

To use this code, clone the repo and launch jupyter lab. Each notebook installs its requirements in the first cell (however in the case of installation issues, you can use AWS SageMaker, e.g. the code was tested with the conda_amazonei_tensorflow2_p36 kernel).

#### Subfolders

* **main_results**: code for modelling consolidation as teacher-student learning, with a modern Hopfield network teacher (representating the initial hippocampal encoding), and a VAE student (representing the generative model trained by consolidation). Also includes code for testing novelty-mediated consolidation, and code for testing how both the teacher and student network could contribute to recall of a partially consolidated memory.
* **additional_distortions**: other memory distortion results (e.g. modelling the Bartlett (1932) and Deese-Roediger-McDermott results)
* **shapes_vae**: code for exploring the latent space of a VAE trained on the shapes3d dataset, and testing how latent representations may support semantic memory.
* **sequence_learning**: code for exploring how the results may extend to memory for sequences (including text and paths on a graph), and how consolidation may improve structural inference.

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
Figure 13 | [./main_results/Generate_novelty_plots.ipynb](./novelty_and_memory/Generate_novelty_plots.ipynb)
Figure 14 | [./main_results/Hybrid_recall.ipynb](./hybrid_recall/Hybrid_recall.ipynb)
Figure 15 | [./sequence_learning/Durrant_sequence_model.ipynb](./sequence_learning/Durrant_sequence_model.ipynb)


