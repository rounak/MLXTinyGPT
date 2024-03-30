# Introduction
This is a Swift implementation of Andrej Karpathy's excellent ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video. It uses the [MLX-Swift](https://github.com/ml-explore/mlx-swift) framework.

The code is all in a single Swift file (< 300 LOC) that implements the Transformer architecture, and uses it to generate Shakespeare like text. 

# How to run
Clone the project, open `MLXTinyGPT.xcodeproj` and run it. It will train the network first, and then generate text.

# Performance
Karpathy's scaled up network acheived a validation loss of 1.4873. On my M3 MBP with 18GB of Memory, I was able to achieve a validation loss of 1.5579436 with a slightly scaled down network, but same the number of epochs.

Karpathy said:
> "I would not run this on a CPU or Macbook. You'll have to break down the number of layers and the embedding dimension and so on".

He's definitely right, but it's cool to see that MLX's unified memory model and Apple's Silicon allows you to get quite close to GPU performance.

# Improvements
1. Trying to scale up parameters like `blockSize` or `nEmbedPerHead` started giving me issues like nan weights or Metal errors. Figure out if there's a way to scale up further without hitting these issues
2. Try to MLX.compile the training loop step. On the initial try, I hit some C++ exceptions, presumably because I wasn't capturing the right set of inputs.
3. Better [tokenization](https://www.youtube.com/watch?v=zduSFxRajkE&t=781s)?
4. Try to save the trained model

# Sample generation
Here's some Shakespeare text the model generated:
> VIRGHARD III:
Come to thee, king we would I would seee with
That bandire, and say I will take with pray him,
Thus hat not husband two any you.
> 
> MARIANA:
Together lies than new's,
Which she's untim's not atch hidle the bred;
Leavefore I long me persing to her precedly.
> 
> ISABELLA:
I he disquanger and 
