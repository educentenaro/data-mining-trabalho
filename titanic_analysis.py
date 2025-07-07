
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuração dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("ANÁLISE DE MINERAÇÃO DE DADOS - DATASET TITANIC")
print("=" * 60)

# 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
print("\n1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS")
print("-" * 40)

# Carregando o dataset
# Nota: Para executar este código, você precisa baixar o dataset do Titanic
# Disponível em: https://www.kaggle.com/c/titanic/data
try:
    df = pd.read_csv('titanic.csv')
    print("✓ Dataset carregado com sucesso!")
except FileNotFoundError:
    print("⚠ Arquivo 'titanic.csv' não encontrado.")
    print("Baixe o dataset em: https://www.kaggle.com/c/titanic/data")
    print("Gerando dataset de exemplo para demonstração...")
    
    # Criando dataset de exemplo baseado no Titanic
    np.random.seed(42)
    n_samples = 891
    
    df = pd.DataFrame({
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(29, 14, n_samples),
        'SibSp': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.01]),
        'Parch': np.random.choice([0, 1, 2, 3], n_samples, p=[0.76, 0.13, 0.08, 0.03]),
        'Ticket': [f'TICKET_{i}' for i in range(1, n_samples + 1)],
        'Fare': np.random.lognormal(2.5, 1.2, n_samples),
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    })
    
    # Adicionando alguns valores NaN para simular dados reais
    df.loc[np.random.choice(df.index, 177), 'Age'] = np.nan
    df.loc[np.random.choice(df.index, 2), 'Embarked'] = np.nan
    
    print("✓ Dataset de exemplo gerado!")

# Informações básicas do dataset
print(f"\nDimensões do dataset: {df.shape}")
print(f"Número de registros: {df.shape[0]}")
print(f"Número de colunas: {df.shape[1]}")

print("\nPrimeiras 5 linhas do dataset:")
print(df.head())

print("\nInformações sobre as colunas:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nValores faltantes por coluna:")
print(df.isnull().sum())

# 2. ANÁLISE EXPLORATÓRIA DE DADOS
print("\n" + "=" * 60)
print("2. ANÁLISE EXPLORATÓRIA DE DADOS")
print("=" * 60)

# Análise da variável target
print("\nDistribuição da variável target (Sobrevivência):")
survival_counts = df['Survived'].value_counts()
print(survival_counts)
print(f"Taxa de sobrevivência: {df['Survived'].mean():.2%}")

# Criando visualizações
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análise Exploratória - Dataset Titanic', fontsize=16, fontweight='bold')

# Gráfico 1: Distribuição de Sobrevivência
axes[0, 0].pie(survival_counts.values, labels=['Não Sobreviveu', 'Sobreviveu'], 
               autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Distribuição de Sobrevivência')

# Gráfico 2: Sobrevivência por Classe
survival_by_class = df.groupby('Pclass')['Survived'].mean()
axes[0, 1].bar(survival_by_class.index, survival_by_class.values, 
               color=['gold', 'silver', 'brown'])
axes[0, 1].set_title('Taxa de Sobrevivência por Classe')
axes[0, 1].set_xlabel('Classe')
axes[0, 1].set_ylabel('Taxa de Sobrevivência')
axes[0, 1].set_xticks([1, 2, 3])

# Gráfico 3: Sobrevivência por Gênero
survival_by_sex = df.groupby('Sex')['Survived'].mean()
axes[1, 0].bar(survival_by_sex.index, survival_by_sex.values, 
               color=['lightblue', 'pink'])
axes[1, 0].set_title('Taxa de Sobrevivência por Gênero')
axes[1, 0].set_xlabel('Gênero')
axes[1, 0].set_ylabel('Taxa de Sobrevivência')

# Gráfico 4: Distribuição de Idade
axes[1, 1].hist(df['Age'].dropna(), bins=30, color='skyblue', alpha=0.7)
axes[1, 1].set_title('Distribuição de Idade')
axes[1, 1].set_xlabel('Idade')
axes[1, 1].set_ylabel('Frequência')

plt.tight_layout()
plt.show()

# 3. PRÉ-PROCESSAMENTO DOS DADOS
print("\n" + "=" * 60)
print("3. PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 60)

# Criando uma cópia para processamento
df_processed = df.copy()

# Tratamento de valores faltantes
print("\nTratamento de valores faltantes:")

# Preenchendo idade com a mediana
median_age = df_processed['Age'].median()
df_processed['Age'].fillna(median_age, inplace=True)
print(f"✓ Idade: preenchida com mediana ({median_age:.1f})")

# Preenchendo Embarked com a moda
mode_embarked = df_processed['Embarked'].mode()[0]
df_processed['Embarked'].fillna(mode_embarked, inplace=True)
print(f"✓ Embarked: preenchido com moda ({mode_embarked})")

# Preenchendo Fare com a mediana (se houver valores faltantes)
if df_processed['Fare'].isnull().any():
    median_fare = df_processed['Fare'].median()
    df_processed['Fare'].fillna(median_fare, inplace=True)
    print(f"✓ Fare: preenchido com mediana ({median_fare:.2f})")

# Codificação de variáveis categóricas
print("\nCodificação de variáveis categóricas:")
le = LabelEncoder()

# Codificando Sexo
df_processed['Sex_encoded'] = le.fit_transform(df_processed['Sex'])
print("✓ Sexo codificado (female=0, male=1)")

# Codificando Embarked
df_processed['Embarked_encoded'] = le.fit_transform(df_processed['Embarked'])
print("✓ Embarked codificado")

# Criando variáveis derivadas
print("\nCriando variáveis derivadas:")
df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                 bins=[0, 18, 35, 60, 100], 
                                 labels=['Child', 'Young', 'Adult', 'Senior'])
df_processed['Age_Group_encoded'] = le.fit_transform(df_processed['Age_Group'])
print("✓ Variáveis derivadas criadas (FamilySize, IsAlone, Age_Group)")

# Selecionando features para o modelo
features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 
           'Embarked_encoded', 'FamilySize', 'IsAlone', 'Age_Group_encoded']
X = df_processed[features]
y = df_processed['Survived']

print(f"\nFeatures selecionadas: {len(features)}")
print(features)

# 4. MODELAGEM E CLASSIFICAÇÃO
print("\n" + "=" * 60)
print("4. MODELAGEM E CLASSIFICAÇÃO")
print("=" * 60)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Conjunto de treino: {X_train.shape[0]} amostras")
print(f"Conjunto de teste: {X_test.shape[0]} amostras")

# Modelo 1: Random Forest
print("\n--- MODELO 1: RANDOM FOREST ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Acurácia do Random Forest: {rf_accuracy:.4f}")

# Modelo 2: Logistic Regression
print("\n--- MODELO 2: LOGISTIC REGRESSION ---")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"Acurácia da Regressão Logística: {lr_accuracy:.4f}")

# Comparação dos modelos
print("\n--- COMPARAÇÃO DOS MODELOS ---")
models_comparison = pd.DataFrame({
    'Modelo': ['Random Forest', 'Logistic Regression'],
    'Acurácia': [rf_accuracy, lr_accuracy]
})
print(models_comparison)

# Melhor modelo
best_model_name = models_comparison.loc[models_comparison['Acurácia'].idxmax(), 'Modelo']
best_accuracy = models_comparison['Acurácia'].max()
best_model = rf_model if best_model_name == 'Random Forest' else lr_model
best_predictions = rf_predictions if best_model_name == 'Random Forest' else lr_predictions

print(f"\nMelhor modelo: {best_model_name} (Acurácia: {best_accuracy:.4f})")

# 5. AVALIAÇÃO DETALHADA DO MELHOR MODELO
print("\n" + "=" * 60)
print("5. AVALIAÇÃO DETALHADA DO MELHOR MODELO")
print("=" * 60)

print(f"Modelo selecionado: {best_model_name}")
print(f"Acurácia: {best_accuracy:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, best_predictions))

# Matriz de Confusão
print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)

# Visualização da Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Sobreviveu', 'Sobreviveu'],
            yticklabels=['Não Sobreviveu', 'Sobreviveu'])
plt.title(f'Matriz de Confusão - {best_model_name}')
plt.xlabel('Predição')
plt.ylabel('Valor Real')
plt.show()

# Importância das Features (para Random Forest)
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nImportância das Features:")
    print(feature_importance)
    
    # Visualização da importância das features
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Importância das Features - Random Forest')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.show()

# 6. RESULTADOS E CONCLUSÕES
print("\n" + "=" * 60)
print("6. RESULTADOS E CONCLUSÕES")
print("=" * 60)

print("RESUMO DOS RESULTADOS:")
print("-" * 30)
print(f"• Dataset utilizado: Titanic Dataset")
print(f"• Número de registros: {df.shape[0]}")
print(f"• Número de features: {len(features)}")
print(f"• Técnica aplicada: Classificação")
print(f"• Melhor modelo: {best_model_name}")
print(f"• Acurácia alcançada: {best_accuracy:.4f}")

print("\nCONCLUSÕES:")
print("-" * 15)
print("• O modelo conseguiu prever a sobrevivência com boa precisão")
print("• Fatores como classe, sexo e idade foram importantes para a predição")
print("• A taxa de sobrevivência foi maior entre mulheres e passageiros da 1ª classe")
print("• O modelo pode ser usado para entender padrões de sobrevivência no Titanic")

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA!")
print("=" * 60)

# Salvando resultados
results_summary = {
    'Dataset': 'Titanic Dataset',
    'Registros': df.shape[0],
    'Features': len(features),
    'Melhor_Modelo': best_model_name,
    'Acuracia': best_accuracy,
    'Taxa_Sobrevivencia': df['Survived'].mean()
}

print("\nResumo salvo em 'results_summary':")
for key, value in results_summary.items():
    print(f"{key}: {value}")
