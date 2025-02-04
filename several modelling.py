import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import linregress
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import threading
from tqdm import tqdm

# Define the file path for the Excel file
file_path = r'c:\Users\Goitsemang\Documents\Python\accounting-app\pbg bank statements\Book1.xlsx'

def load_excel_file(file_path, progress_bar):
    """Function to load the Excel file and update the progress bar."""
    try:
        for i in range(100):
            progress_bar['value'] += 1
            root.update_idletasks()
        
        df = pd.read_excel(file_path)
        progress_bar.stop()
        progress_bar.destroy()
        print("Excel file loaded successfully.")
        print(df.head())
        df_info = df.info()
        print(df_info)
        return df
    except Exception as e:
        progress_bar.stop()
        progress_bar.destroy()
        print(f"Error loading the Excel file: {e}")
        return None

def show_result_in_window(result_text, append=False):
    """Function to display the result in a new window with typing and blinking effects."""
    root.after(0, _show_result_in_window, result_text, append)

def _show_result_in_window(result_text, append):
    result_window = tk.Toplevel()
    result_window.title("Result")
    result_window.configure(bg='#f0f0f0')

    result_label = tk.Label(result_window, text="", font=("Montserrat", 12), bg='#f0f0f0', fg='green')
    result_label.pack(pady=20, padx=20)
    
    def on_close():
        result_window.destroy()
        root.deiconify()

    result_window.protocol("WM_DELETE_WINDOW", on_close)
    
    def type_text(index=0):
        if index < len(result_text):
            current_text = result_label.cget("text")
            if append:
                result_label.config(text=current_text + result_text[index])
            else:
                result_label.config(text=result_text[:index + 1])
            result_window.after(50, type_text, index + 1)
        else:
            result_window.after(5000, stop_blink, result_label)

    def stop_blink(label):
        label.config(fg='green')
    
    def blink():
        current_color = result_label.cget("foreground")
        next_color = "green" if current_color == "#f0f0f0" else "#f0f0f0"
        result_label.config(fg=next_color)
        result_window.after(500, blink)
    
    type_text()
    blink()

def generate_weekly_statement():
    """Function to generate a weekly statement of transactions."""
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()

    def generate_statement():
        if df is not None:
            df['Date'] = pd.to_datetime(df['Date'])
            weekly_df = df.set_index('Date').resample('W').sum()
            weekly_statement_text = weekly_df.to_string()
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("Weekly Statement:\n" + weekly_statement_text)
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.")

    threading.Thread(target=generate_statement).start()

def generate_monthly_statement():
    """Function to generate a monthly statement of transactions."""
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()

    def generate_statement():
        if df is not None:
            df['Date'] = pd.to_datetime(df['Date'])
            # Sum all amounts of the month
            monthly_sum = df.set_index('Date').resample('MS')['Amount'].sum()
            monthly_statement_text = monthly_sum.to_string()
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("Monthly Summary of Amounts:\n" + monthly_statement_text)
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.")

    threading.Thread(target=generate_statement).start()

def plot_pairplot():
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()
    
    def plot():
        if df is not None:
            pairplot = sns.pairplot(df)
            pairplot.fig.suptitle('Pairplot of Numeric Columns', y=1.02)
            plt.show()
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("Pairplot of Numeric Columns has been displayed.")
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.")
    
    threading.Thread(target=plot).start()

def linear_regression_analysis():
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()
    
    def analyze():
        if df is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    slope, intercept, r_value, p_value, std_err = linregress(df[col1], df[col2])
                    print(f'Linear Regression between {col1} and {col2}:')
                    print(f'Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}, P-value: {p_value}, Std Err: {std_err}')
                    
                    plt.figure(figsize=(10, 6))
                    sns.regplot(x=col1, y=col2, data=df, line_kws={"color": "red"}, scatter_kws={'alpha':0.5})
                    plt.title(f'Linear Regression: {col1} vs {col2}', fontsize=14)
                    plt.xlabel(col1, fontsize=12)
                    plt.ylabel(col2, fontsize=12)
                    plt.grid(True)
                    plt.show()
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("Linear Regression Analysis has been displayed.")
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.")
    
    threading.Thread(target=analyze).start()

def decision_tree_regression():
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()
    
    def regress():
        if df is not None:
            X = df[['Amount']]
            y = df['Balance']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            depths = [3, 5, 7, 9, 11]
            train_errors = []
            test_errors = []

            plt.figure(figsize=(15, 20))

            for i, depth in enumerate(depths):
                model = DecisionTreeRegressor(random_state=42, max_depth=depth)
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                train_errors.append(mean_squared_error(y_train, y_train_pred))
                test_errors.append(mean_squared_error(y_test, y_test_pred))
                
                plt.subplot(len(depths), 1, i+1)
                plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
                plt.scatter(X_test, y_test_pred, color='red', label='Predicted', alpha=0.6)
                plt.xlabel('Amount', fontsize=14)
                plt.ylabel('Balance', fontsize=14)
                plt.title(f'Decision Tree Regression (Depth={depth})', fontsize=16)
                plt.legend()
                plt.grid(True)

            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 8))
            plt.plot(depths, train_errors, label='Training Error', marker='o')
            plt.plot(depths, test_errors, label='Testing Error', marker='o')
            plt.xlabel('Tree Depth', fontsize=14)
            plt.ylabel('Mean Squared Error', fontsize=14)
            plt.title('Training and Testing Errors vs. Tree Depth', fontsize=16)
            plt.legend()
            plt.grid(True)
            plt.show()

            depth_to_plot = 5
            model = DecisionTreeRegressor(random_state=42, max_depth=depth_to_plot)
            model.fit(X_train, y_train)

            plt.figure(figsize=(20, 10))
            plot_tree(model, feature_names=['Amount'], filled=True, rounded=True, fontsize=12)
            plt.title(f'Decision Tree Regressor (Depth={depth_to_plot})', fontsize=16)
            plt.show()

            y_pred = model.predict(X_test)

            plt.figure(figsize=(12, 8))
            plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
            plt.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.6)
            plt.xlabel('Amount', fontsize=14)
            plt.ylabel('Balance', fontsize=14)
            plt.title(f'Decision Tree Regression (Depth={depth_to_plot}): Actual vs Predicted', fontsize=16)
            plt.legend()
            plt.grid(True)
            plt.show()

            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("Decision Tree Regression results have been displayed.")
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.")
    
    threading.Thread(target=regress).start()

def polynomial_regression():
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()
    
    def regress():
        if df is not None:
            X = df[['Amount']]
            y = df['Balance']
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X)

            X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

            poly_model = LinearRegression()
            poly_model.fit(X_poly_train, y_train)

            y_poly_pred = poly_model.predict(X_poly_test)

            poly_mse = mean_squared_error(y_test, y_poly_pred)
            poly_r2 = r2_score(y_test, y_poly_pred)
            print(f"Polynomial Regression Mean Squared Error: {poly_mse}")
            print(f"Polynomial Regression R-squared: {poly_r2}")

            plt.figure(figsize=(12, 8))
            plt.scatter(X, y, color='blue', label='Data', alpha=0.6)
            X_grid = np.arange(min(X.values), max(X.values), 0.1)
            X_grid_poly = poly.transform(X_grid.reshape(-1, 1))
            plt.plot(X_grid, poly_model.predict(X_grid_poly), color='red', label='Polynomial Regression')
            plt.xlabel('Amount', fontsize=14)
            plt.ylabel('Balance', fontsize=14)
            plt.title('Polynomial Regression (Degree 3)', fontsize=16)
            plt.legend()
            plt.grid(True)
            plt.show()
            
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("Polynomial Regression results have been displayed.")
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.")
    
    threading.Thread(target=regress).start()

def logistic_regression():
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()

    def regress():
        if df is not None:
            df['Target'] = (df['Balance'] > 0).astype(int)

            X_class = df[['Amount']]
            y_class = df['Target']

            X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

            log_model = LogisticRegression()
            log_model.fit(X_train_class, y_train_class)

            y_class_pred = log_model.predict(X_test_class)
            y_class_prob = log_model.predict_proba(X_test_class)[:, 1]

            accuracy = accuracy_score(y_test_class, y_class_pred)
            conf_matrix = confusion_matrix(y_test_class, y_class_pred)
            print(f"Logistic Regression Accuracy: {accuracy}")
            print("Confusion Matrix:")
            print(conf_matrix)

            # Plot Confusion Matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

            # Plot ROC Curve
            fpr, tpr, _ = roc_curve(y_test_class, y_class_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()

            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("Logistic Regression results have been displayed.")
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.")

    threading.Thread(target=regress).start()

def random_forest_regression():
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()
    
    def regress():
        if df is not None:
            X = df[['Amount']]
            y = df['Balance']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }

            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

            # Fit the model without progress bar
            grid_search.fit(X_train, y_train)

            print("Best parameters found: ", grid_search.best_params_)
            print("Best score: ", grid_search.best_score_)

            best_rf_model = grid_search.best_estimator_
            y_rf_pred = best_rf_model.predict(X_test)

            rf_mse = mean_squared_error(y_test, y_rf_pred)
            rf_r2 = r2_score(y_test, y_rf_pred)
            print(f"Tuned Random Forest Mean Squared Error: {rf_mse}")
            print(f"Tuned Random Forest R-squared: {rf_r2}")

            cv_scores = cross_val_score(best_rf_model, X, y, cv=5)
            print("Cross-validation scores: ", cv_scores)
            print("Mean cross-validation score: ", np.mean(cv_scores))

            plt.figure(figsize=(12, 8))
            plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
            plt.scatter(X_test, y_rf_pred, color='red', label='Predicted', alpha=0.6)
            plt.xlabel('Amount', fontsize=14)
            plt.ylabel('Balance', fontsize=14)
            plt.title('Random Forest Regression: Actual vs Predicted', fontsize=16)
            plt.legend()
            plt.grid(True)
            plt.show()

            feature_imp = pd.Series(best_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print("Feature Importances:")
            print(feature_imp)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_imp, y=feature_imp.index)
            plt.xlabel('Feature Importance Score', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.title('Visualizing Important Features', fontsize=16)
            plt.show()

            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("Random Forest Regression results have been displayed.")
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.")

    threading.Thread(target=regress).start()
def kmeans_clustering():
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()
    
    def cluster():
        if df is not None:
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df[['Amount', 'Balance']])

            plt.figure(figsize=(12, 8))
            sns.scatterplot(x='Amount', y='Balance', hue='Cluster', data=df, palette='viridis')
            plt.xlabel('Amount', fontsize=14)
            plt.ylabel('Balance', fontsize=14)
            plt.title('K-Means Clustering', fontsize=16)
            plt.legend()
            plt.grid(True)
            plt.show()
            
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("K-Means Clustering results have been displayed.")
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.")
    
    threading.Thread(target=cluster).start()


def voting_classifier():
    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()
    
    def classify():
        if df is not None:
            df['Target'] = (df['Balance'] > 0).astype(int)
            X_class = df[['Amount']]
            y_class = df['Target']
            X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_class = scaler.fit_transform(X_train_class)
            X_test_class = scaler.transform(X_test_class)

            log_model = LogisticRegression()
            rf_model = RandomForestClassifier(random_state=42)
            svm_model = SVC(probability=True, random_state=42)
            knn_model = KNeighborsClassifier()
            gb_model = GradientBoostingClassifier(random_state=42)
            ada_model = AdaBoostClassifier(random_state=42)

            log_model.fit(X_train_class, y_train_class)
            rf_model.fit(X_train_class, y_train_class)
            svm_model.fit(X_train_class, y_train_class)
            knn_model.fit(X_train_class, y_train_class)
            gb_model.fit(X_train_class, y_train_class)
            ada_model.fit(X_train_class, y_train_class)

            voting_clf = VotingClassifier(estimators=[
                ('lr', log_model), 
                ('rf', rf_model),
                ('svm', svm_model),
                ('knn', knn_model),
                ('gb', gb_model),
                ('ada', ada_model)
            ], voting='soft')

            voting_clf.fit(X_train_class, y_train_class)

            models = {
                'Logistic Regression': log_model,
                'Random Forest': rf_model,
                'SVM': svm_model,
                'KNN': knn_model,
                'Gradient Boosting': gb_model,
                'AdaBoost': ada_model,
                'Voting Classifier': voting_clf
            }

            results = ""
            for name, model in models.items():
                y_pred = model.predict(X_test_class)
                accuracy = accuracy_score(y_test_class, y_pred)
                results += f"{name} Accuracy: {accuracy}\n"
                results += classification_report(y_test_class, y_pred, zero_division=0) + "\n"
                results += "Confusion Matrix:\n"
                results += str(confusion_matrix(y_test_class, y_pred)) + "\n\n"
            
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window(results, append=False)
        else:
            progress_bar.stop()
            progress_bar.destroy()
            show_result_in_window("DataFrame is not loaded.", append=False)

    threading.Thread(target=classify).start()

def show_result_in_window(result_text, append=False):
    """Function to display the result in a new window with typing and blinking effects."""
    root.after(0, _show_result_in_window, result_text, append)

def _show_result_in_window(result_text, append):
    result_window = tk.Toplevel()
    result_window.title("Result")
    result_window.configure(bg='#f0f0f0')

    result_label = tk.Label(result_window, text="", font=("Montserrat", 12), bg='#f0f0f0', fg='green')
    result_label.pack(pady=20, padx=20)
    
    def on_close():
        result_window.destroy()
        root.deiconify()

    result_window.protocol("WM_DELETE_WINDOW", on_close)
    
    def type_text(index=0):
        if index < len(result_text):
            current_text = result_label.cget("text")
            if append:
                result_label.config(text=current_text + result_text[index])
            else:
                result_label.config(text=result_text[:index + 1])
            result_window.after(50, type_text, index + 1)
        else:
            result_window.after(5000, stop_blink, result_label)

    def stop_blink(label):
        label.config(fg='green')
    
    def blink():
        current_color = result_label.cget("foreground")
        next_color = "green" if current_color == "#f0f0f0" else "#f0f0f0"
        result_label.config(fg=next_color)
        result_window.after(500, blink)
    
    type_text()
    result_label.after(0, stop_blink, result_label) # Stop blinking after displaying the text

# Function to create the main menu (assuming the rest of your code is the same)
def create_menu():
    global root
    root = tk.Tk()
    root.title("Menu")
    root.configure(bg='#e6f7ff')

    label = tk.Label(root, text="Select an analysis or model to run:", font=("Helvetica", 14), bg='#e6f7ff')
    label.pack(pady=10)

    choices = [
        "1. Pairplot of Numeric Columns",
        "2. Linear Regression Analysis",
        "3. Decision Tree Regression",
        "4. Polynomial Regression",
        "5. Logistic Regression",
        "6. Random Forest Regression",
        "7. K-Means Clustering",
        "8. Voting Classifier",
        "9. Sort by Description",
        "10. Generate Weekly Statement",
        "11. Generate Monthly Statement",
        "12. Exit"
    ]

    for choice in choices:
        button = tk.Button(root, text=choice, font=("Helvetica", 12), bg='#cceeff', activebackground='#80d4ff', command=lambda c=choice: on_choice(c.split('.')[0]))
        button.pack(pady=5, padx=20, fill='x')

    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()
    
    def load_file():
        global df
        df = load_excel_file(file_path, progress_bar)
    
    threading.Thread(target=load_file).start()

    root.mainloop()

# Run the menu
create_menu()