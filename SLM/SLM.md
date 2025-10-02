# SLM - Small Language Model

This notebook dives deep into the core concepts of Small Language Models (SLMs) and demonstrates how they work **from scratch** using the [Tiny Stories](https://arxiv.org/pdf/2305.07759) dataset.

We'll walk through every step of building a minimal yet functional language model, covering:

* Tokenization and dataset preprocessing
* Building a simple neural network for language modeling
* Training loop and loss optimization
* Generating new text (stories) using the trained model
* Exploring limitations and possible improvements

The **Tiny Stories** dataset, introduced in the paper *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, is a lightweight and well-curated corpus of short, simple English stories. It is ideal for training and evaluating small-scale models due to its clean structure and limited vocabulary.

**Inspired by:** [Vizuara’s TinyStories SLM Tutorial](https://youtu.be/pOFcwcwtv3k?si=6vUqLaowEID3IFa3)
This notebook builds upon the concepts demonstrated in the tutorial, expanding them with additional explanations, experiments, and insights.