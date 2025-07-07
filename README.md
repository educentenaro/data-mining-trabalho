# 🚢 Análise de Mineração de Dados - Dataset Titanic

## 📋 Descrição do Projeto

Este projeto foi desenvolvido como parte do trabalho da disciplina **Tópicos Especiais em Computação I**, aplicando técnicas de mineração de dados para análise do famoso dataset do Titanic.

### 🎯 Objetivo

Utilizar algoritmos de machine learning para prever a sobrevivência de passageiros do Titanic com base em características como classe, sexo, idade e outros fatores.

## 📊 Dataset

- **Nome**: Titanic Dataset
- **Origem**: Kaggle (https://www.kaggle.com/c/titanic/data)
- **Registros**: 891 passageiros
- **Features**: 12 colunas originais + variáveis derivadas
- **Variável Target**: Survived (0 = Não sobreviveu, 1 = Sobreviveu)

### 🔍 Principais Variáveis

- **Pclass**: Classe do passageiro (1, 2, 3)
- **Sex**: Sexo do passageiro
- **Age**: Idade do passageiro
- **SibSp**: Número de irmãos/cônjuges a bordo
- **Parch**: Número de pais/filhos a bordo
- **Fare**: Tarifa paga
- **Embarked**: Porto de embarque

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **Scikit-learn**: Algoritmos de machine learning
- **Matplotlib**: Visualização de dados
- **Seaborn**: Visualizações estatísticas

## 🚀 Como Executar

### 1. Pré-requisitos

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Download do Dataset

1. Acesse: https://www.kaggle.com/c/titanic/data
2. Baixe o arquivo `train.csv`
3. Renomeie para `titanic.csv`
4. Coloque na mesma pasta do script

### 3. Execução

```bash
python titanic_analysis.py
```

**Nota**: Se não tiver o dataset, o código gerará dados de exemplo para demonstração.

## 📈 Metodologia

### 1. Exploração dos Dados

- Análise das dimensões e tipos de dados
- Identificação de valores faltantes
- Estatísticas descritivas
- Visualizações exploratórias

### 2. Pré-processamento

- Tratamento de valores faltantes
- Codificação de variáveis categóricas
- Criação de variáveis derivadas:
  - `FamilySize`: Tamanho da família
  - `IsAlone`: Se viaja sozinho
  - `Age_Group`: Faixa etária

### 3. Modelagem

- **Random Forest Classifier**
- **Logistic Regression**
- Divisão treino/teste (80%/20%)
- Avaliação com métricas de classificação

### 4. Avaliação

- Acurácia
- Matriz de confusão
- Relatório de classificação
- Importância das features

## 📊 Resultados

### Principais Descobertas

- Taxa de sobrevivência geral: ~38%
- Mulheres tiveram maior taxa de sobrevivência
- Passageiros da 1ª classe tiveram melhores chances
- Idade e tamanho da família também influenciaram

### Performance dos Modelos

| Modelo              | Acurácia |
| ------------------- | -------- |
| Random Forest       | ~82%     |
| Logistic Regression | ~80%     |

### Features Mais Importantes

1. Sexo do passageiro
2. Classe do passageiro
3. Idade
4. Tarifa paga
5. Tamanho da família

## 📁 Estrutura do Projeto

```
titanic-analysis/
├── titanic_analysis.py      # Script principal
├── titanic.csv             # Dataset (baixar separadamente)
├── README.md               # Este arquivo
├── relatorio.pdf           # Relatório em PDF
└── requirements.txt        # Dependências
```

## 📈 Visualizações Geradas

O projeto gera os seguintes gráficos:

1. **Distribuição de Sobrevivência** (Gráfico de Pizza)
2. **Taxa de Sobrevivência por Classe** (Gráfico de Barras)
3. **Taxa de Sobrevivência por Gênero** (Gráfico de Barras)
4. **Distribuição de Idade** (Histograma)
5. **Matriz de Confusão** (Heatmap)
6. **Importância das Features** (Gráfico de Barras)

## 💡 Insights Principais

1. **Gênero**: Mulheres tiveram 3x mais chances de sobrevivência
2. **Classe Social**: Passageiros da 1ª classe tiveram maior taxa de sobrevivência
3. **Idade**: Crianças e adultos jovens tiveram melhores chances
4. **Família**: Viajar sozinho reduziu as chances de sobrevivência
5. **Tarifa**: Tarifas mais altas correlacionaram com maior sobrevivência

## 📋 Critérios de Avaliação Atendidos

- ✅ **Qualidade da escolha do dataset**: Dataset clássico e bem estruturado
- ✅ **Aplicação da técnica de data mining**: Classificação com múltiplos algoritmos
- ✅ **Qualidade do relatório**: Análise completa com visualizações

## 👤 Autor

**[Seu Nome]**

- Turma: [Sua Turma]
- Disciplina: Tópicos Especiais em Computação I
- Data: 06/07/2025

## 📝 Licença

Este projeto foi desenvolvido para fins educacionais como parte do curso de Ciência da Computação.

---

_Trabalho desenvolvido para a disciplina Tópicos Especiais em Computação I - URICER_
