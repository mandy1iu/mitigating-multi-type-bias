import argparse
import os
import numpy as np
import torch
import transformers
import pandas as pd
from bias_bench.dataset import load_inlp_data
from bias_bench.debias.inlp import compute_projection_matrix
# from bias_bench.debias.inlp import before_compute_projection_matrix
from bias_bench.debias.inlp.context_nullspace_projection import before_compute_projection_matrix

from bias_bench.model import models
from bias_bench.util import generate_experiment_id

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Computes the projection matrix for INLP.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertModel",
    choices=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"],
    help="Model (e.g., BertModel) to compute the INLP projection matrix for. "
    "Typically, these correspond to a HuggingFace class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    default="gender",
    choices=["gender", "race", "religion"],
    help="What type of bias to compute the INLP projection matrix for.",
)
parser.add_argument(
    "--n_classifiers",
    action="store",
    type=int,
    default=80,
    help="Number of classifiers to train when computing projection matrix.",
)
parser.add_argument("--seed", action="store", type=int, default=0, help="Seed for RNG.")


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="projection",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        seed=args.seed,
    )

    print("Computing projection matrix:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - n_classifiers: {args.n_classifiers}")
    print(f" - seed: {args.seed}")

    # Load data for INLP classifiers.
    # data = load_inlp_data(args.persistent_dir, args.bias_type, seed=args.seed)
    data_gender = load_inlp_data(args.persistent_dir, "gender", seed=args.seed)
    data_race = load_inlp_data(args.persistent_dir, "race", seed=args.seed)
    data_religion = load_inlp_data(args.persistent_dir, "religion", seed=args.seed)

    # df = pd.DataFrame(data_gender)
    df = pd.DataFrame(data_race)
    # df = pd.DataFrame(data_religion)

    # Define the path where you want to save the CSV file
    # csv_file_path = "gender_data8.csv"
    csv_file_path = "race_data2.csv"
    # csv_file_path = "religion_data2.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")



    # print(data_gender)

    # Load model and tokenizer.
    model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # three data in sequence
    X_train1, X_dev1, X_test1, Y_train1, Y_dev1, Y_test1 = before_compute_projection_matrix(
        model,tokenizer,data_gender,bias_type="gender")
    X_train2, X_dev2, X_test2, Y_train2, Y_dev2, Y_test2 = before_compute_projection_matrix(
        model,tokenizer,data_race,bias_type="race")
    X_train3, X_dev3, X_test3, Y_train3, Y_dev3, Y_test3 = before_compute_projection_matrix(
        model,tokenizer,data_religion,bias_type="religion")

    # # label method2: Y{-1, 0, 1}--{2, 3, 4}
    # Y_train1 = Y_train1 + 3
    # Y_dev1 = Y_dev1 + 3
    # Y_test1 = Y_test1 + 3

    # Y_train3 = Y_train3 + 5
    # Y_dev3 = Y_dev3 + 5
    # Y_test3 = Y_test3 + 5


    X_train = np.concatenate((X_train1, X_train2, X_train3), axis=0)
    X_dev = np.concatenate((X_dev1, X_dev2, X_dev3), axis=0)
    X_test = np.concatenate((X_test1, X_test2, X_test3), axis=0)

    Y_train = np.concatenate((Y_train1, Y_train2, Y_train3), axis=0)
    Y_dev = np.concatenate((Y_dev1, Y_dev2, Y_dev3), axis=0)
    Y_test = np.concatenate((Y_test1, Y_test2, Y_test3), axis=0)


    # print(X_train)
    # print(Y_train)


    # 获取乱序索引
    train_indices = np.random.permutation(len(X_train))
    dev_indices = np.random.permutation(len(X_dev))
    test_indices = np.random.permutation(len(X_test))

    # 对数据进行乱序  
    X_train = X_train[train_indices]
    X_dev = X_dev[dev_indices]
    X_test = X_test[test_indices]

    Y_train = Y_train[train_indices]
    Y_dev = Y_dev[dev_indices]
    Y_test = Y_test[test_indices]

    # projection_matrix = compute_projection_matrix(
    #     model,
    #     tokenizer,
    #     data,
    #     bias_type=args.bias_type,
    #     n_classifiers=args.n_classifiers,
    # )
    projection_matrix = compute_projection_matrix(X_train, X_dev, X_test, Y_train, Y_dev, Y_test,n_classifiers=args.n_classifiers)


    print(
        f"Saving computed projection matrix to: {args.persistent_dir}/results/projection_matrix/{experiment_id}.pt"
    )
    os.makedirs(f"{args.persistent_dir}/results/projection_matrix", exist_ok=True)
    torch.save(
        projection_matrix,
        f"{args.persistent_dir}/results/projection_matrix/{experiment_id}.pt",
    )

