# Plaintext Recovery from DES Encrypted Data Using Bidirectional LSTM

## 🎯 Project Overview

This project explores how deep learning techniques, specifically Bidirectional Long Short-Term Memory (BiLSTM) networks, can be applied to recover plaintext from DES (Data Encryption Standard) encrypted ciphertext. The work involves generating datasets, training deep learning models, and analyzing how well the model could learn patterns from encrypted data across multiple encryption rounds, all without access to the encryption key.

This project combines interests in cryptography, deep learning, and cybersecurity, providing valuable insights into the challenges of using AI techniques for cryptanalysis.

## 📁 Project Structure

```
DES-Decryption-Using-BiLSTM/
├── scripts/
│   ├── generate_data.py      # Data generation script for DES encryption/decryption
│   ├── train.py              # Training script for BiLSTM model
│   ├── test.py               # Testing script for model evaluation
│   ├── final_train.py        # Final training implementation
│   └── final_test.py         # Final testing implementation
├── data/                     # Dataset storage
├── models/                   # Trained model checkpoints
├── output/                   # Output files and results
├── notebooks/                # Jupyter notebooks for analysis
└── README.md                 # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Suraj-G-Rao/DES-Decryption-Using-BiLSTM.git
cd DES-Decryption-Using-BiLSTM
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data models output notebooks
```

## 📊 Usage

### Data Generation

Generate DES encrypted datasets for training:

```bash
python scripts/generate_data.py
```

This script creates plaintext-ciphertext pairs with varying numbers of DES encryption rounds.

### Model Training

Train the BiLSTM model on the generated dataset:

```bash
python scripts/train.py
```

For the final training implementation:

```bash
python scripts/final_train.py
```

### Model Testing

Evaluate the trained model's performance:

```bash
python scripts/test.py
```

For the final testing implementation:

```bash
python scripts/final_test.py
```

## 🔬 Methodology

### Data Generation
- Generate random plaintext messages
- Apply DES encryption with varying numbers of rounds
- Create plaintext-ciphertext pairs for training

### Model Architecture
- **Bidirectional LSTM**: Processes sequences in both forward and backward directions
- **Multiple Layers**: Deep architecture for complex pattern learning
- **Attention Mechanism**: Focuses on relevant parts of the ciphertext

### Training Process
- Split dataset into training, validation, and test sets
- Train on various encryption rounds to assess learning capability
- Monitor loss and accuracy metrics

## 📈 Results

The project demonstrates:
- Model's ability to learn patterns in encrypted data
- Performance analysis across different encryption rounds
- Insights into the feasibility of AI-based cryptanalysis
- Limitations and challenges in plaintext recovery without keys

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **TensorFlow/PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Data visualization

## 🙏 Acknowledgments

I'm extremely grateful to my mentors for their guidance and continuous support throughout this project:

- **Prof. Alwyn Roshan Pais** - Mentor and Guide
- **Dr. Purushothama B R** - Mentor and Guide

Their expertise and encouragement were invaluable in shaping this research and helping me navigate the complex intersection of deep learning and cryptography.

## 🔮 Future Work

This experience has been a rewarding step in my learning journey, and I look forward to exploring more intersections of machine learning and information security in the future, including:

- Exploring other encryption algorithms
- Investigating different neural network architectures
- Developing more sophisticated cryptanalysis techniques
- Contributing to the field of AI-based security research

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any improvements or suggestions.

---

*This project was developed as part of an internship focused on exploring the applications of deep learning in cryptography and cybersecurity.*