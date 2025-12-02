import torch
from lstm.model import LSTMModel
from lstm.tokenizer import build_vocab
from lstm.generator import generate_smiles

def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training or generating SMILES using LSTM.

    Returns:
        argparse.Namespace containing:
            train_path (str): Path to training SMILES file.
            val_path (str): Path to validation SMILES file.
            tokenizer_vocab_path (str): Vocabulary-building SMILES file.
            model_path (str): Where to save/load model checkpoint.
            hidden_dim (int): LSTM hidden dimension.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout between layers.
            num_epochs (int): Max training epochs.
            batch_size (int): Training batch size.
            learning_rate (float): Learning rate.
            patience (int): Early stopping patience.
            start_token (str): For generation mode.
            max_length (int): Max generated sequence length.
            max_attempts (int): Max generation retries.
    """
    parser = argparse.ArgumentParser(description="Train or generate SMILES with LSTM")

    # Data and vocabulary
    parser.add_argument("--train_path", type=str, default="data/train.txt")
    parser.add_argument("--val_path", type=str, default="data/val.txt")
    parser.add_argument("--tokenizer_vocab_path", type=str, default="data/smiles.csv")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")

    # Model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)

    # Training settings
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)

    # Generation arguments
    parser.add_argument("--start_token", type=str, default="C")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=10)

    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabulary
    char_to_int, int_to_char = build_vocab(args.tokenizer_vocab_path)

    input_dim = len(char_to_int)
    output_dim = len(char_to_int)

    # Initialize model and load weights
    model = LSTMModel(input_dim, args.hidden_dim, output_dim, args.num_layers, args.dropout).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Generate SMILES
    smiles = generate_smiles(
        model=model,
        char_to_int=char_to_int,
        int_to_char=int_to_char,
        start_token=args.start_token,
        max_length=args.max_length,
        max_attempts=args.max_attempts
    )

    if smiles:
        print("Generated valid SMILES:", smiles)
    else:
        print("Failed to generate a valid SMILES.")

if __name__ == "__main__":
    main()
