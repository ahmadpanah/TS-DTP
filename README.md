# Task-Specific Dynamic Token Pruning (TS-DTP) for LLMs

This repository contains the code and resources for our research paper on Task-Specific Dynamic Token Pruning (TS-DTP) for Large Language Models (LLMs). TS-DTP is a novel approach designed to adapt token pruning to the specific demands of downstream NLP tasks, enhancing both efficiency and, in some cases, performance.

## Overview

Large Language Models (LLMs) have achieved remarkable performance across a wide range of Natural Language Processing (NLP) tasks. However, their computational demands, particularly during inference, can be a major barrier to deployment, especially in resource-constrained environments. Dynamic token pruning, which selectively removes less crucial tokens during inference, has been proposed as a promising solution to reduce this computational burden. 

Our work introduces Task-Specific Dynamic Token Pruning (TS-DTP), which builds upon the concept of dynamic token pruning but enhances it by incorporating task-specific information. This is achieved through task-specific attention mechanisms and feature representations, allowing the pruning strategy to be tailored to the specific requirements of each downstream task, ensuring the retention of the most relevant tokens. This adaptation is key to not only improving efficiency but also maintaining or improving performance on downstream tasks.

## Key Contributions

Our research makes the following key contributions:

1.  **TS-DTP Framework:** We present TS-DTP, a novel framework that extends the concept of dynamic token pruning to incorporate task-specific information, enabling finer-grained, more efficient optimization for LLMs.
2.  **Task-Specific Attention:** We introduce a methodology for incorporating task-specific attention into the pruning mechanism, allowing different tokens to be selected based on task requirements.
3.  **Empirical Validation:** We demonstrate the effectiveness of TS-DTP through comprehensive empirical evaluations on diverse downstream NLP tasks, including sentiment analysis (SST-2), question answering (QNLI), and machine translation (WMT16).
4.  **Improved Efficiency and Performance:** Our results show that TS-DTP achieves notable reductions in computational cost (inference speed, memory usage) and, in some instances, improves the performance of LLMs on these tasks compared to generic, task-agnostic pruning approaches.
5.  **Comprehensive Analysis:** We analyze the effects of token pruning on the model's performance and resource utilization, aiming to understand how task-specific pruning affects the model's efficiency and capability.

## Code Implementation

The core components of our implementation are in the `TS-DTP.ipynb` file.

### `TaskSpecificAttention` Class

This module computes task-specific attention weights based on the input hidden states and task-specific feature representations.

*   Initializes a task-specific weight matrix (`task_specific_weight`) and a feature layer to compute the task-specific features.
*   The `forward` method computes the task-specific attention based on these parameters.

### `TaskSpecificDynamicTokenPruning` Class

This class implements the core TS-DTP framework.

*   It loads the pre-trained model and tokenizer.
*   It defines a list of `TaskSpecificAttention` modules for each layer.
*   `calculate_token_importance` calculates the cumulative importance scores for tokens.
*   `prune_tokens` implements a dynamic thresholding mechanism to prune tokens.
*   The `forward` method combines task-specific attention with standard attention, prunes tokens, and then performs classification using a linear layer.
*   The `calculate_auxiliary_loss` computes an auxiliary loss term for the training.

### Example Code

The `if __name__ == '__main__':` block shows an example of how to train and evaluate TS-DTP.  It:

*   Sets up the pre-trained model, tokenizer, and the training configuration.
*   Loads data from the SST-2 dataset (using the `datasets` library from Hugging Face).
*   Implements a realistic training and validation loop.
*   Calculates and reports the validation accuracy.

### Dataset Class

A `SST2Dataset` class provides the data loading and preparation for the sentiment analysis task, which includes tokenization, padding and truncation. This class should be extended for other tasks according to the task input requirements.

## Usage

1.  **Install Dependencies:**
    ```bash
    pip install torch transformers datasets scikit-learn tqdm
    ```
2.  **Clone the Repository:**
    ```bash
    git clone [[repository_link]](https://github.com/ahmadpanah/TS-DTP)
    cd [TS-DTP]
    ```
3.  **Run the Code:**
    ```bash
    python TS-DTP.ipynb
    ```
    Adjust the model and dataset parameters in the `if __name__ == '__main__':` section according to your needs.

## Experimental Results

The experimental evaluation, detailed in our paper, uses three datasets:
*  **SST-2:** a dataset for sentiment classification
*  **QNLI:** a question answering dataset
*   **WMT16 English-German:** a dataset for machine translation

Our approach, TS-DTP, is compared to several baselines:
*   **No Pruning:** Fine-tuning without any token pruning.
*   **General DTP:** Applying the baseline dynamic token pruning strategy (Keith et al., 2024).
*   **Fine-Tuned Model:** Fine-tuning without dynamic token pruning.

TS-DTP achieves competitive accuracy (or BLEU scores) with reduced computational cost (improved speed and memory consumption) on all datasets.

## Future Work

Future research directions include:

*   Adaptive hyperparameter tuning for TS-DTP based on input data and task specifics.
*   Extending TS-DTP to other LLM architectures and multimodal LLMs.
*   Exploring how to generalize TS-DTP across tasks, possibly using meta-learning.


## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact: h.ahmadpanah@iau.ac.ir
