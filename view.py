import streamlit as st
import pandas as pd
from model import NaiveBayes, DecisionTree, RandomForest, EvaluationMetrics

from sklearn.model_selection import train_test_split


import streamlit as st
import pandas as pd

import time


import numpy as np
def main():
    st.title("Data Mining Model Evaluation")
    st.subheader("Perbandingan Algortima Decision Tree (C4.5), Random Forest, dan Naive")

    uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        # Memeriksa apakah 'Label' ada di kolom
        if 'Label' in df.columns:
            st.write("Dataset Preview:")
            st.write(df.head())

            # Definisikan kolom 'Label' sebagai kelas
            class_col = 'Label'
            y = df[class_col]
            X = df.drop(columns=[class_col])

            st.write("Info Dataset:")
            st.write(f"Jumlah Baris: {len(df)}")
            st.write(f"Jumlah Kolom: {len(df.columns)}")

            # Input split data
            split_ratio = st.sidebar.slider("Training-Testing Split Ratio", 0.1, 0.9, 0.3)
            st.write(f"Pengguna memilih split ratio: {split_ratio}")

  
            # Input jumlah K-Fold
            num_splits = st.sidebar.number_input("Number of K-Fold Splits", min_value=2, max_value=10, value=5)
            st.write(f"Pengguna memilih jumlah K-Fold: {num_splits}")

            # Memeriksa apakah sudah dipilih split dan fold
            if split_ratio and num_splits:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
                st.write(f"Jumlah data training: {len(X_train)}")
                st.write(f"Jumlah data testing: {len(X_test)}")

            if st.button("Tes Naive Bayes"):
                start_time = time.time()
                naive_bayes = NaiveBayes()
                eval_metrics = EvaluationMetrics()
                accuracies, confusion_matrices, execution_time = eval_metrics.evaluate_model(naive_bayes, X, y, split_ratio, num_splits)
                end_time = time.time()
                st.write(f"Waktu Eksekusi Naive Bayes: {end_time - start_time:.2f} detik")

                st.write("Akurasi Rata-rata Naive Bayes:", eval_metrics.average_accuracy)
                st.write("Confusion Matrix Keseluruhan Naive Bayes:")
                st.write(eval_metrics.overall_confusion_matrix)

                st.write("Akurasi Naive Bayes per Fold:")
                for i, acc in enumerate(accuracies):
                    st.write(f"Fold {i+1}: {acc}")
                st.write("Confusion Matrix Naive Bayes per Fold:")
                for i, cm in enumerate(confusion_matrices):
                    st.write(f"Fold {i+1}:")
                    st.write(cm)


            # Tombol tes Decision Tree
            if st.button("Tes Decision Tree"):
                start_time = time.time()
                decision_tree = DecisionTree()
                eval_metrics = EvaluationMetrics()
                accuracies, confusion_matrices, execution_time = eval_metrics.evaluate_model(decision_tree, X, y, split_ratio, num_splits)
                end_time = time.time()
                st.write(f"Waktu Eksekusi Decision Tree: {end_time - start_time:.2f} detik")
                st.write("Akurasi Rata-rata Decision Tree:", eval_metrics.average_accuracy)
                st.write("Confusion Matrix Keseluruhan Decision Tree:")
                st.write(eval_metrics.overall_confusion_matrix)

                st.write("Akurasi Decision Tree per Fold:")
                for i, acc in enumerate(accuracies):
                    st.write(f"Fold {i+1}: {acc}")
                st.write("Confusion Matrix Decision Tree per Fold:")
                for i, cm in enumerate(confusion_matrices):
                    st.write(f"Fold {i+1}:")
                    st.write(cm)


            # Tombol tes Random Forest
            if st.button("Tes Random Forest"):
                start_time = time.time()
                random_forest = RandomForest()
                eval_metrics = EvaluationMetrics()
                accuracies, confusion_matrices, execution_time = eval_metrics.evaluate_model(random_forest, X, y, split_ratio, num_splits)
                end_time = time.time()
                st.write(f"Waktu Eksekusi Random Forest: {end_time - start_time:.2f} detik")

                st.write("Akurasi Rata-rata Random Forest:", eval_metrics.average_accuracy)
                st.write("Confusion Matrix Keseluruhan Random Forest:")
                st.write(eval_metrics.overall_confusion_matrix)

                st.write("Akurasi Random Forest per Fold:")
                for i, acc in enumerate(accuracies):
                    st.write(f"Fold {i+1}: {acc}")
                st.write("Confusion Matrix Random Forest per Fold:")
                for i, cm in enumerate(confusion_matrices):
                    st.write(f"Fold {i+1}:")
                    st.write(cm)
            
            if st.button("Perbandingan"):
                start_time = time.time()
                naive_bayes = NaiveBayes()
                decision_tree = DecisionTree()
                random_forest = RandomForest()

                eval_metrics_nb = EvaluationMetrics()
                eval_metrics_dt = EvaluationMetrics()
                eval_metrics_rf = EvaluationMetrics()

                accuracies_nb, _, execution_time_nb = eval_metrics_nb.evaluate_model(naive_bayes, X, y, split_ratio, num_splits)
                accuracies_dt, _, execution_time_dt = eval_metrics_dt.evaluate_model(decision_tree, X, y, split_ratio, num_splits)
                accuracies_rf, _, execution_time_rf = eval_metrics_rf.evaluate_model(random_forest, X, y, split_ratio, num_splits)

                end_time = time.time()

                df_comparison = pd.DataFrame({
                    "Model": ["Naive Bayes", "Decision Tree", "Random Forest"],
                    "Akurasi Rata-rata": [
                        np.mean(accuracies_nb),
                        np.mean(accuracies_dt),
                        np.mean(accuracies_rf)
                    ],
                    "Waktu Eksekusi (detik)": [
                        execution_time_nb,
                        execution_time_dt,
                        execution_time_rf
                    ]
                })

                st.write("Hasil Perbandingan Model:")
                st.write(df_comparison)

            

        else:
            st.write("Mohon pilih file dengan kolom 'Label' untuk menentukan kelas.")
