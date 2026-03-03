# gpt_from_scratch

A small project where I re-implemented a GPT-style Transformer from
scratch to truly understand what is happening under the hood (thanks Andrej).

Some comments are in Spanish, some are written mid-thought, etc...

------------------------------------------------------------------------

## What this is

A minimal character-level language model that grows step by step:

-   Bigram baseline\
-   Token + positional embeddings\
-   Causal self-attention\
-   Multi-head attention\
-   Feed-forward layers\
-   Stacked Transformer blocks\
-   Autoregressive generation

Everything is implemented explicitly: masking, residual connections,
layer norm, projection layers.

------------------------------------------------------------------------

## Core idea

Instead of treating Transformers as a black box, this repo builds the
mechanism piece by piece:

-   Why divide by sqrt(d)?
-   Why multiple heads?
-   Why residual streams?
-   Why pre-norm stabilises training?

------------------------------------------------------------------------

## Structure

. ├── bigram.py \# full implementation + training loop\
├── gpt-dev.ipynb \# notebook version for experiments\
├── input.txt \# training corpus

------------------------------------------------------------------------

## Run

python bigram.py

The script trains and then generates text from a zero context. Note that I don't train it for too long but, if you have the resources (and time to write checkpoints, make the GPU nodes talk, etc.), feel free to scale it up.
